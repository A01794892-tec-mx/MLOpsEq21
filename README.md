# MLOPs Team Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOPs Team Project

## Team Member
JOSUÉ PÉREZ 						A00752186

LINETH GUERRA 						A01795639

JOSE GONZALEZ 						A01795755

EDUARDO RODRIGUEZ 					A01794892

OMAR JUÁREZ 						A01795499

JESÚS MARTÍNEZ 						A01740049

## Project Setup

### GIT

```
git clone https://github.com/A01794892-tec-mx/MLOpsEq21.git
```

### CMD (Windows) 

Open a CMD as Administrator

```
cd <directorio_local>
python -m venv MLOpsEq21_venv
.\MLOpsEq21_venv\Scripts\activate
pip install -r requirements.txt
cd mlops
docker-compose --env-file config.env up -d --build
```
## Reproducibility Guidelines

### Repeatable Splitting
Use the params.yaml file to define the seed to avoid repeating it across different files.

### Workflow Pipeline
Always use pipeline execution to ensure the correct sequence of steps:

```
python ./src/mlops/mlopsPipe.py --config ./params.yaml
```

### Data Versioning
The information currently monitored for changes with DVC includes:

```
./data
./models
./reports
```

Add a file or folder to version control:

```
dvc add <file_or_folder_path>
```

Commit with a change:
```
dvc commit -m "commit message"
```

Push changes:
```
dvc push
```

Pull changes:
```
dvc pull
```

### Model Versioning
The models are stored in:

```
./models
```

The models are versioned using the same DVC commands mentioned previously.

### Environment Versioning
The text file requirements.txt contains all the dependencies used by our application

The Dockerfile contains all the necessary instructions for environment setup, as well as the subsequent installation of dependencies, to ultimately run the application and expose it via Docker networking.

## Team 21 - Structure Reference

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

