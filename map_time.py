import os
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path

import dateutil.parser
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
import typer
import unidecode
from PIL import Image
from sodapy import Socrata
from tqdm import tqdm

import utils

mapbox_token = os.getenv("MAPBOX_TOKEN")


def main(
    database_id: str = "gt2j-8ykr",
    use_cache: bool = False,
    location_file: str = "s3://www.charlielito.ml/data/location.csv",
):
    df, location_df = utils.get_data(database_id, location_file, use_cache)

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

    def get_scatter_mapbox(df, size=40):
        return go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            marker=go.scattermapbox.Marker(
                size=size * fn(df.cases) / max_cases, color="red"
            ),
            text=[f"{row.location} -> {row.cases} casos" for i, row in gb.iterrows()],
            hoverinfo="text",
        )

    df["fecha_de_notificaci_n"] = df.fecha_de_notificaci_n.apply(
        dateutil.parser.isoparse
    )
    date_start = np.min(df.fecha_de_notificaci_n.values)
    date_start = datetime.utcfromtimestamp(date_start.astype(int) * 1e-9)
    date_end = np.max(df.fecha_de_notificaci_n.values)
    date_end = datetime.utcfromtimestamp(date_end.astype(int) * 1e-9)
    num_days = (date_end - date_start).days

    date = date_start
    frames = []
    for day_delta in tqdm(range(0, num_days + 1, 1), total=num_days):
        date = date_start + timedelta(days=day_delta)
        tmp = df[df.fecha_de_notificaci_n <= date]
        gb_tmp = utils.get_groupby_location(tmp, location_df)

        total = sum(gb_tmp.cases)
        date_str = f"{utils.MONTHS_LOOKUP[date.month]} {date.day} {date.year}"

        curr_fig = go.Figure(
            data=[get_scatter_mapbox(gb_tmp)],
            layout=dict(
                title={"text": f"Total casos: {total} <br>" f"{date_str}",},
                font=dict(family="Courier New, monospace", size=20, color="#7f7f7f"),
                mapbox=dict(
                    style="dark",
                    accesstoken=mapbox_token,
                    zoom=3.8,
                    center=go.layout.mapbox.Center(lat=center.lat, lon=center.lon),
                ),
                showlegend=False,
            ),
        )

        image_bytes = curr_fig.to_image(format="png")
        image = utils.bytes2image(image_bytes)
        frames.append(Image.fromarray(image))

    frames[0].save(
        "images/animation.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=60,
        loop=0,
    )


if __name__ == "__main__":
    typer.run(main)
