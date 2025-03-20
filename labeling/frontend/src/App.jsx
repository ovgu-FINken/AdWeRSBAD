import { useState, useEffect } from "react";
import { fetchLocations, fetchWeathers, fetchImage, postWeatherLabel, setLocation } from "./api";

function App() {
  const [locations, setLocations] = useState([]);
  const [weathers, setWeathers] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [imageLabel, setImageLabel] = useState(null);
  const [labeledImageCount, setLabeledImageCount] = useState(0); // New state for labeled image count
  const [selectedLabel, setSelectedLabel] = useState(null);
  const [labelCount, setLabelCount] = useState(0);

  // Fetch locations when the app starts
  useEffect(() => {
    const loadLocations = async () => {
      const locs = await fetchLocations();
      setLocations(locs);
    };
    loadLocations();
  }, []); 

  // Fetch weathers when the app starts
  useEffect(() => {
    const loadWeathers = async () => {
      const weathers = await fetchWeathers();
      setWeathers(weathers);
    };
    loadWeathers();
  }, []); 

  // Handle location selection
  const handleLocationSelect = async (location) => {
    console.log("setting locaiton:", location);
    setSelectedLocation(location);
    await setLocation(location); // Send selected location to backend
    setImageData(null); // Clear the current image when location changes
    setImageId(null); // Reset image ID
    setImageLabel(null); // Reset image label
    setSelectedLabel(null); // Reset selected label
  };

  // Load next image based on selected location
  const loadNextImage = async () => {
    if (!selectedLocation) return;
    
    const data = await fetchImage();  // Pass location as argument to the backend API
    if (data) {
      setImageData(`data:image/jpeg;base64,${data.image}`);
      setImageId(data.id);
      setImageLabel(data.label);
      setSelectedLabel(null);
    }
  };

  // Label the image and send update to backend
  const labelWeather = async (label) => {
    if (!imageId) return;
    setSelectedLabel(label);

    const response = await postWeatherLabel(imageId, label);
    if (response) {
      console.log("Label successfully saved:", response);
      setLabeledImageCount(response.labeled_images); 
      loadNextImage();
    }
  };

  // useEffect(() => {
  //   const eventSource = new EventSource("/api/update_label");
  //   eventSource.onmessage = function (event) {
  //     const labeledImages = parseInt(event.data, 10);
  //     setLabelCount(labeledImages);  // Update the label count
  //   };
  //   eventSource.onerror = function () {
  //     console.error("Error in event stream");
  //   };
  //   return () => {
  //     eventSource.close();  // Clean up the event source when the component unmounts
  //   };
  // }, []);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100vh",
      width: "100vw",
      backgroundColor: "#121212",
      color: "#fff",
      fontFamily: "Arial, sans-serif",
      padding: "20px"
    }}>
      <h1>Weather Labeling Tool</h1>

      {/* Select Location */}
      {!selectedLocation ? (
        <div style={{ textAlign: "center" }}>
          <h2>Select a Location</h2>
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: "15px",
            marginTop: "15px"
          }}>
            {locations.map((loc) => (
              <button
                key={loc}
                onClick={() => handleLocationSelect(loc)} // Use the handleLocationSelect function
                style={{
                  padding: "15px 30px",
                  fontSize: "16px",
                  fontWeight: "bold",
                  backgroundColor: "#444",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer",
                  transition: "background 0.3s",
                  minWidth: "200px"
                }}
              >
                {loc}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          {/* Show selected location */}
          <h2>Selected Location: {selectedLocation}</h2>
            {/* Display the counter for labeled images */}
            {labeledImageCount > 0 && (
              <p style={{ color: "green", fontSize: "18px" }}>
                {labeledImageCount} image{labeledImageCount > 1 ? "s" : ""} labeled successfully.
              </p>
            )}
          {/* Button to load next image */}
          <button
            onClick={loadNextImage}  // Make sure this calls the correct function
            style={{
              padding: "15px 30px",
              fontSize: "18px",
              backgroundColor: "#333",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
              marginBottom: "20px",
              transition: "background 0.3s"
            }}
          >
            Load Next Image
          </button>
          {/* Show number of labeled images */}
          {labeledImageCount > 0 && (
            <p style={{ color: "green", fontSize: "18px" }}>
              {labeledImageCount} image{labeledImageCount > 1 ? "s" : ""} labeled successfully.
            </p>
          )}
          {/* Show image if available */}
          {imageData ? (
            <>
              <img src={imageData} alt="Weather Sample" style={{ maxWidth: "90%", maxHeight: "60vh", borderRadius: "10px" }} />
              <p><strong>Database Label:</strong> {imageLabel || "None"}</p>
              <p><strong>Selected Label:</strong> {selectedLabel || "None"}</p>
            </>
          ) : (
            <p>No image loaded.</p>
          )}

          {/* Weather Labeling Buttons */}
          <div style={{ 
            display: "flex", 
            flexWrap: "wrap", 
            justifyContent: "center", 
            gap: "15px", 
            marginTop: "15px",
            width: "100%"
          }}>
            {weathers.map((label) => (
              <button
                key={label}
                onClick={() => labelWeather(label)}
                style={{
                  padding: "15px 30px",
                  fontSize: "16px",
                  fontWeight: "bold",
                  backgroundColor: selectedLabel === label ? "#007bff" : "#444",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer",
                  transition: "background 0.3s",
                  minWidth: "150px"
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
