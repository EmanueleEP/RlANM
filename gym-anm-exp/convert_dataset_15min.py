"""Converti CSV o JSON PVGIS in un CSV con campioni ogni 15 minuti.

Uso: python convert_dataset_15min.py INPUT [-o OUTPUT]
Richiede pandas. Le unità e i segni dei valori rimangono invariati.
Per CSV diversi da PQSf e T1, usare --time-column e --time-format.
I timestamp Unix sono interpretati in UTC; le date senza fuso lo conservano
non specificato. Le colonne testuali (per esempio localtime) sono escluse.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_dataset(path, time_column=None, time_format=None):
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as file:
            data = json.load(file)
        frame = pd.DataFrame(data["outputs"]["hourly"])
        column = time_column or "time"
        timestamps = pd.to_datetime(
            frame[column], format=time_format or "%Y%m%d:%H%M", utc=True
        )
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, encoding="utf-8-sig")
        column = time_column or next(
            (name for name in ("Timestamp", "Date/Time", "timestamp", "time", "datetime")
             if name in frame.columns), None
        )
        if column is None:
            raise ValueError("Colonna temporale non riconosciuta: specificare --time-column.")
        if pd.api.types.is_numeric_dtype(frame[column]) and time_format is None:
            timestamps = pd.to_datetime(frame[column], unit="s", utc=True)
        else:
            date_format = time_format or ("%d %m %Y %H:%M" if column == "Date/Time" else None)
            timestamps = pd.to_datetime(frame[column], format=date_format)
    else:
        raise ValueError("Formato supportato: CSV oppure JSON PVGIS.")

    if timestamps.isna().any():
        raise ValueError("Il dataset contiene timestamp mancanti.")
    values = frame.drop(columns=[column]).select_dtypes(include="number")
    if values.empty:
        raise ValueError("Nessuna colonna numerica trovata.")
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.isna().all().any():
        raise ValueError("Una colonna numerica non contiene valori validi.")
    values.index = pd.DatetimeIndex(timestamps)
    # I campioni duplicati allo stesso istante vengono mediati.
    return values.groupby(level=0).mean().sort_index()


def convert(values):
    if len(values) < 2:
        raise ValueError("Servono almeno due timestamp distinti.")
    interval = pd.Timedelta(minutes=15)
    typical_step = values.index.to_series().diff().median()
    if typical_step < interval:
        # Media nei quarti d'ora [00:00, 00:15), [00:15, 00:30), ecc.
        result = values.resample("15min").mean()
        result = result.interpolate(method="time", limit_area="inside")
    else:
        # Conserva gli istanti originali nell'interpolazione, anche se non
        # coincidono con i quarti d'ora (per esempio PVGIS alle hh:10).
        target = pd.date_range(values.index.min().ceil("15min"),
                               values.index.max().floor("15min"), freq="15min")
        result = values.reindex(values.index.union(target))
        result = result.interpolate(method="time", limit_area="inside").reindex(target)
    if result.empty or result.isna().any().any():
        raise ValueError("Dati insufficienti per ottenere una serie completa a 15 minuti.")
    result.index.name = "timestamp"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="CSV oppure JSON PVGIS")
    parser.add_argument("-o", "--output", type=Path, help="CSV di destinazione")
    parser.add_argument("--time-column", help="Nome della colonna temporale")
    parser.add_argument("--time-format", help="Formato delle date, es. %%d/%%m/%%Y %%H:%%M")
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_15min.csv")
    if output.suffix.lower() != ".csv":
        parser.error("Il file di destinazione deve avere estensione .csv.")
    try:
        values = read_dataset(args.input, args.time_column, args.time_format)
        result = convert(values)
        # Non sovrascrivere il dataset originale o risultati già presenti.
        with output.open("x", encoding="utf-8", newline="") as file:
            result.to_csv(file)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Errore: {error}\n")
    print(f"Creato {output}: {len(values)} campioni originali → {len(result)} a 15 minuti.")
    print("Valori interpolati anche negli intervalli mancanti interni; unità e segni invariati.")


if __name__ == "__main__":
    main()
