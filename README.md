# CO2 Emission Calculator & Sustainability Analytics

A modular, feature-rich Streamlit web application for calculating, visualizing, and optimizing CO2 emissions across supply chains, warehousing, packaging, and more.

## Features
- **CO2 Emissions Calculator**: Calculate emissions for shipments by route, mode, and weight.
- **Route Visualizer**: Interactive map of emission routes and route optimization.
- **Supplier Lookup**: Filter and analyze suppliers by country, city, material, and green score.
- **Reports**: Emission summaries, route optimization, and CSV export.
- **Optimized Route Planning**: Find the most eco-friendly transport route.
- **Green Warehousing**: Analyze savings from green technologies in warehouses.
- **Sustainable Packaging**: Assess packaging emissions and alternatives.
- **Carbon Offsetting**: Plan and record offset projects.
- **Efficient Load Management**: Optimize logistics for fewer trips and lower emissions.
- **Energy Conservation**: Analyze facility energy savings from smart systems.

## Tech Stack
- [Streamlit](https://streamlit.io/) (UI)
- [Pandas](https://pandas.pydata.org/) (Data)
- [Folium](https://python-visualization.github.io/folium/) (Maps)
- [Plotly](https://plotly.com/python/) (Charts)
- [SQLite](https://www.sqlite.org/index.html) (Database)
- [PyYAML](https://pyyaml.org/) (Config)
- [Geopy](https://geopy.readthedocs.io/) (Geocoding)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/C360.git
   cd C360
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root with the following variables:
   ```
   # API Keys (required for full functionality)
   WEATHER_API_KEY=your_openweathermap_api_key_here
   ORS_API_KEY=your_openrouteservice_api_key_here
   MARINETRAFFIC_API_KEY=your_marinetraffic_api_key_here

   # Database Configuration
   DB_DIR=./data
   MAX_DB_CONNECTIONS=10
   DB_CONNECTION_TIMEOUT=30

   # Application Settings
   DEBUG=False
   LOG_LEVEL=INFO
   MAX_CONNECTIONS=10
   REQUEST_TIMEOUT=5
   MAX_RETRIES=3

   # Cache Settings
   CACHE_TTL=3600
   MAX_CACHE_SIZE=1000

   # Rate Limiting
   MIN_REQUEST_INTERVAL=0.1
   MAX_REQUESTS_PER_MINUTE=60

   # Security
   ALLOWED_HOSTS=localhost,127.0.0.1
   SECRET_KEY=your_secret_key_here
   ```

   You can get API keys from:
   - OpenWeatherMap: https://openweathermap.org/api
   - OpenRouteService: https://openrouteservice.org/
   - MarineTraffic: https://www.marinetraffic.com/en/api

3. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Initialize the database:**
   ```bash
   python -c "import db; db.init_db()"
   ```

5. **Run the app:**
   ```bash
   python -m streamlit run main.py
   ```

## Security Notes
- Never commit your `.env` file to version control
- Keep your API keys secure and don't share them
- The application will work with limited functionality if API keys are not provided
- All API calls are rate-limited and cached to prevent abuse
- Database connections are pooled and have timeouts
- Input validation is performed on all user inputs
- SQL queries use parameterized statements to prevent injection

## File Structure
```
C360/
├── main.py                # Streamlit entry point
├── db.py                  # Database operations
├── emissions.py           # Emissions and optimization logic
├── visualization.py       # Map and chart helpers
├── ui.py                  # UI helpers (tooltips, errors, etc.)
├── utils.py               # Shared utilities (caching, localization)
├── config.yaml            # App configuration and constants
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── tests/
    └── test_emissions.py  # Unit test stubs
```

## Error Handling
The application includes comprehensive error handling for:
- Database operations (with connection pooling and timeouts)
- API calls (with retries and rate limiting)
- Input validation
- File operations
- Network issues
- Resource cleanup

## Logging
- Logs are written to both console and file (`app.log`)
- Log level can be configured via environment variable
- Different log levels for different components
- Rotating log files to prevent disk space issues

## Contribution Guidelines
- Fork the repo and create a feature branch.
- Add/modify code in a modular way (see file structure).
- Add or update tests in the `tests/` folder.
- Submit a pull request with a clear description.

## License
MIT License 

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue if needed

## Acknowledgments

- OpenRouteService for routing data
- OpenWeatherMap for weather data
- MarineTraffic for maritime data
- Streamlit for the web interface
- Folium for map visualization 