import axios from "axios";

// Axios instance
const api = axios.create({
  baseURL: "http://127.0.0.1:8000", // Ensure this is correct
  timeout: 30000,
});

// Fetch available locations
export const fetchLocations = async () => {
  try {
    const response = await api.get("/api/get_locations");
    return response.data.locations;
  } catch (error) {
    console.error("Error fetching locations:", error);
    return [];
  }
};

// Fetch available weathers
export const fetchWeathers = async () => {
  try {
    const response = await api.get("/api/get_weathers");
    return response.data.weathers;
  } catch (error) {
    console.error("Error fetching weathers:", error);
    return [];
  }
};
// Set the location (initializes dataset in backend)
export const setLocation = async (location) => {
  try {
    await api.post("/api/set_location", JSON.stringify({location}), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error setting location:", error);
    console.error("sent:", JSON.stringify({location}));
  }
};

// Fetch an image (location is already stored in backend)
export const fetchImage = async () => {
  try {
    const response = await api.get("/api/get_image");
    return response.data;
  } catch (error) {
    if (error.response && error.response.status === 400) {
      console.error("Error fetching image: Dataset not initialized. Please select a location.");
    } else {
      console.error("Error fetching image:", error);
    }
    return null;
  }
};

// Post weather label
export const postWeatherLabel = async (imageId, label) => {
  try {
    const response = await api.post("/api/update_label", {
      image_id: imageId, // Must match FastAPI's expected request format
      label: label,
    });
    return response.data; // Expecting { message: "Label updated successfully" }
  } catch (error) {
    console.error("Error posting label:", error);
    return null;
  }
};

