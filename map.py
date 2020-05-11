import os
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
import typer
import unidecode
from sodapy import Socrata

import utils

mapbox_token = os.getenv("MAPBOX_TOKEN")
api_key = os.getenv("DATA_KEY")


def main(
    database_id: str = "gt2j-8ykr",
    use_cache: bool = False,
    # location_file: str = "location.csv",
    location_file: str = "s3://www.charlielito.ml/data/location.csv",
    viz: bool = False,
    output_html: str = "s3://www.charlielito.ml/index.html",
):
    client = Socrata("www.datos.gov.co", api_key)
    results = client.get(database_id, limit=10000)

    df = pd.DataFrame.from_records(results)
    df["location"] = df.apply(
        lambda row: f"{row.ciudad_de_ubicaci_n} {row.departamento}", axis=1
    )
    # remove accents
    df["location"] = df.location.apply(lambda x: str(unidecode.unidecode(x)).upper())
    cities = pd.unique(df.location)

    print("Checking location file...")
    if use_cache and utils.file_exists(location_file):
        print("File exists now reading")
        location_df = pd.read_csv(location_file)
        print("Done")

    else:
        print("Could not locate file, calculating from scratch")
        utils.maybe_mkdirs(location_file)

        print("Calculating...")
        location_df = utils.get_location_df(cities)
        print("Ready! Writing to remote...")
        location_df.to_csv(location_file, index=False)
        print("Done")

    print("Check if cities matches")
    if len(location_df.location.values) != len(cities):
        new_cities = set(cities) - set(location_df.location.values)
        print("Difference", new_cities)
        print("Calculating new cities")
        _location_df = utils.get_location_df(list(new_cities))
        location_df = location_df.append(_location_df)
        print("Done!")

        # save updated version
        print("Ready! Writing to remote...")
        location_df.to_csv(location_file, index=False)
        print("Done")

    gb = utils.get_groupby_location(df, location_df)
    center = location_df[location_df.location == "BOGOTA BOGOTA D.C."].iloc[0]

    fn = lambda x: np.log(x / 6 + 1)
    max_cases = fn(max(gb.cases))
    total = sum(gb.cases)
    print(f"TOTAL CASES: {total}",)

    print("Getting plotly figure...")
    utc_today = pytz.utc.localize(datetime.utcnow())
    col_today = utc_today.astimezone(pytz.timezone("America/Bogota"))
    today_str = col_today.strftime("%Y-%m-%d %I:%M %p")

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=gb["lat"],
                lon=gb["lon"],
                marker=go.scattermapbox.Marker(
                    size=40 * fn(gb.cases) / max_cases, color="red"
                ),
                text=[
                    f"{row.location} -> {row.cases} casos" for i, row in gb.iterrows()
                ],
                hoverinfo="text",
            ),
        ],
        layout=dict(
            title={
                "text": f"Casos COVID-19 en Colombia: {total} total <br>"
                f"Actualizado: {today_str}",
                "xanchor": "center",
                "x": 0.5,
                # "yanchor": "top",
                # "y": 0.95,
            },
            font=dict(family="Courier New, monospace", size=24, color="#7f7f7f"),
            mapbox=dict(
                style="dark",
                accesstoken=mapbox_token,
                zoom=5,
                center=go.layout.mapbox.Center(lat=center.lat, lon=center.lon),
            ),
            showlegend=False,
            hoverlabel=dict(font=dict(size=25)),
        ),
    )
    if viz:
        fig.show()

    print("Done!!")
    with StringIO() as str_buf:
        fig.write_html(str_buf)
        out_str = str_buf.getvalue()

    print("Writing html to s3")
    utils.write_file(output_html, out_str)
    print("Done")


if __name__ == "__main__":
    typer.run(main)