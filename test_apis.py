import os
import openrouteservice
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Debug: Print current working directory and environment variables
logging.info(f"Current working directory: {os.getcwd()}")
logging.info("Environment variables:")
for key in ['ORS_API_KEY']:
    value = os.getenv(key)
    if value:
        masked_value = value[:4] + '*' * (len(value) - 4)
        logging.info(f"{key}: {masked_value}")
    else:
        logging.info(f"{key}: Not found")

def test_ors_api():
    """Test OpenRouteService API key."""
    api_key = os.getenv('ORS_API_KEY')
    if not api_key:
        logging.error("ORS API key not found in environment variables")
        return False
    
    try:
        # Initialize ORS client
        client = openrouteservice.Client(key=api_key)
        
        # Test with a simple route in Paris
        coords = [[2.3522, 48.8566], [2.3522, 48.8606]]  # Paris coordinates
        route = client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        
        if route and 'features' in route and len(route['features']) > 0:
            distance = route['features'][0]['properties']['segments'][0]['distance']
            logging.info(f"ORS API test successful. Route distance: {distance/1000:.2f} km")
            return True
        else:
            logging.error("ORS API test failed. No route data received")
            return False
            
    except Exception as e:
        logging.error(f"ORS API test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    logging.info("Testing ORS API key...")
    ors_success = test_ors_api()
    if ors_success:
        logging.info("ORS API test passed successfully!")
    else:
        logging.error("ORS API test failed. Please check the logs above for details.") 