```
tableai/
├── pyproject.toml             # Project configuration and dependencies
├── README.md                  # Project documentation
├── tableai/                   # Main package
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Core utilities and abstractions
│   │   ├── __init__.py        # Export core components
│   │   ├── abstractions.py    # Abstract classes and interfaces
│   │   ├── document.py        # Document-related models and utilities
│   │   ├── database.py        # Database utilities
│   │   ├── path.py            # Path management utilities
│   │   ├── table.py           # Table extraction and models
│   │   ├── time_util.py       # Time utilities
│   │   └── services.py        # Core services implementation
│   ├── api/                   # FastAPI implementation
│   │   ├── __init__.py        # API package initialization
│   │   ├── app.py             # FastAPI application definition
│   │   ├── models.py          # Pydantic models for API
│   │   ├── document_routes.py # Document-related endpoints
│   │   ├── table_routes.py    # Table-related endpoints
│   │   └── integration_routes.py # Integration endpoints
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py         # Logging configuration
│   │   ├── validation.py      # Input validation
│   │   └── export.py          # Data export utilities
│   ├── integrations/          # External integrations
│   │   ├── __init__.py
│   │   ├── dropbox.py         # Dropbox integration
│   │   └── google_drive.py    # Google Drive integration
│   └── config.py              # Configuration management
├── frontend/                  # React frontend application
│   ├── package.json           # Frontend dependencies
│   ├── public/                # Static assets
│   └── src/                   # React source code
│       ├── components/        # Reusable UI components
│       │   ├── Header.jsx
│       │   ├── DocumentList.jsx
│       │   ├── TableViewer.jsx
│       │   └── FileUpload.jsx
│       ├── pages/             # Page components
│       │   ├── Dashboard.jsx
│       │   ├── DocumentDetail.jsx
│       │   ├── TableDetail.jsx
│       │   └── Settings.jsx
│       ├── services/          # API service functions
│       │   ├── api.js         # API client
│       │   ├── documentService.js
│       │   └── tableService.js
│       ├── contexts/          # React contexts
│       │   └── AppContext.jsx
│       ├── App.jsx            # Main application component
│       └── index.jsx          # Application entry point
├── tests/                     # Test suite
│   ├── conftest.py            # Test configuration
│   ├── test_core/             # Core tests
│   │   ├── test_document.py
│   │   └── test_table.py
│   ├── test_api/              # API tests
│   │   ├── test_document_routes.py
│   │   └── test_table_routes.py
│   └── test_integrations/     # Integration tests
│       └── test_dropbox.py
└── scripts/                   # Utility scripts
    ├── setup_dev.sh           # Development environment setup
    └── run_tests.sh           # Test runner script
```