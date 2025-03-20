import base64
import datetime
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from torchvision import tv_tensors
from torchvision.utils import save_image

# Assume your database class provides an iterator
from adwersbad import Adwersbad
from adwersbad.db_helpers import get_connection

app = FastAPI()

# List of origins allowed to make requests to the API
origins = [
    "http://localhost:5173",  # Your frontend's address (Vite dev server)
    "http://127.0.0.1:5173",  # Another possible localhost variant
]

# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
ds = None
current_location = None
last_label = None
last_timestamp = None


class LabelUpdate(BaseModel):
    image_id: int
    label: str


def tensor_to_base64(tensor: tv_tensors.Image) -> str:
    img_byte_arr = BytesIO()
    save_image(tensor, img_byte_arr, format="jpeg")  # Save image as JPEG
    img_str = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    return img_str


@app.get("/api/get_locations")
def get_locations():
    """Fetch all distinct locations from the weather table."""
    with get_connection("psycopg@local") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT location FROM weather;")
            locations = [row[0] for row in cur.fetchall()]
    return JSONResponse({"locations": locations})


@app.get("/api/get_weathers")
def get_weathers():
    """Fetch all distinct weathers from the weather table."""
    # with get_connection("psycopg@local") as conn:
    #     with conn.cursor() as cur:
    #         cur.execute("SELECT DISTINCT weather FROM weather;")
    #         locations = [row[0] for row in cur.fetchall()]
    weathers = [
        "Clear Sky",
        "Mainly Clear",
        "Overcast",
        "Wet",
        "Light Rain",
        "Heavy Rain",
        "Snow",
        "Snowing",
    ]
    return JSONResponse({"weathers": weathers})


@app.post("/api/set_location")
async def set_location(request: Request):
    # Read the raw body as string
    body = await request.body()
    print(f"Raw body received: {body.decode()}")  # Print raw body for debugging
    try:
        data = await request.json()  # Try to parse JSON
        location = data.get("location")
        if location:
            print(f"Location received: {location}")
        else:
            print("No location in request body.")
    except Exception as e:
        print(f"Error parsing JSON: {e}")

    global ds, current_location
    if location == current_location:
        print("dataset already initialized")
        return {"message": f"Dataset already set for {location}"}
    current_location = location
    print(f"Initializing dataset for location: {location}")
    ds = Adwersbad(
        data={
            "weather": ["weather_uid", "time", "location", "weather", "custom"],
            "camera": ["image"],
        },
        orderby="time",
        location=location,
    )
    print(f"images: {ds.count}")
    ds = iter(ds)
    return {"message": f"Dataset initialized for {location}"}


@app.get("/api/get_image")
def get_image():
    """Fetch an unlabeled image from the dataset."""
    global ds, last_label_timestamp

    if ds is None:
        raise HTTPException(
            status_code=400, detail="Dataset not initialized. Select a location first."
        )
    new_label = 1
    while new_label:
        print(new_label)
        try:
            uid, last_label_timestamp, loc, label, new_label, image = next(ds)
        except StopIteration:
            raise HTTPException(status_code=404, detail="No unlabeled images found.")

    img_str = tensor_to_base64(image)
    return JSONResponse(
        {
            "image": img_str,
            "id": uid,
            "label": label,
        }
    )


@app.post("/api/update_label")
def update_label(data: LabelUpdate):
    """Update the weather label for an image."""
    global last_label_timestamp
    upds = Adwersbad(
        data={"weather": ["weather_uid", "time"]},
        orderby="time",
        location=current_location,
    )
    with get_connection("psycopg@local") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE weather SET custom = %s WHERE weather_uid = %s;",
                (data.label, data.image_id),
            )
            labeled_images = 0
            for uid, ts in iter(upds):
                if ts < last_label_timestamp:
                    continue
                if abs(last_label_timestamp - ts) > datetime.timedelta(seconds=60):
                    conn.commit()
                    break
                cur.execute(
                    "UPDATE weather SET custom = %s WHERE weather_uid = %s;",
                    (data.label, uid),
                )
                labeled_images += 1
    return {"message": "Label updated successfully", "labeled_images": labeled_images}
