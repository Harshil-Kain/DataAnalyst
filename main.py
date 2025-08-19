'''
This code is the generalized version of the project Virtual-TA
This is the final version of the code that can be used to generate code for any question related to data analysis.
'''

import os
import re
import time
import subprocess
import sys
import openai
import traceback
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional


load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

RETRIES = 5
MODEL = "gpt-4.1-nano"

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

def ask_llm_to_generate_code(question):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python data analysis expert. Given a user's task, generate a complete Python script. "
                    "The code must:\n"
                    "1. Automatically download or scrape any required dataset from the web (e.g., from Wikipedia).\n"
                    "2. Perform all needed analysis, cleaning, or transformation.\n"
                    "3. Print or return final answers in clear format (e.g., print(), JSON, etc).\n"
                    "4. If visualization is needed, return a Base64-encoded PNG image under 100 KB.\n"
                    "5. You must ensure Pandas code avoids SettingWithCopyWarning by either using .copy() when creating new DataFrames or .loc for assignment.\n"
                    "6. Use the latest version of scikit-learn.\n"
                    "7. If any libraries are missing, install them using pip at runtime.\n"
                    "8. Do not use any external files or resources, everything must be self-contained.\n"
                    "9. Do not use any imports that are not necessary for the task.\n"
                    "10. Only include the images or their links in the output if they are required by the task.\n"
                    "11. If the task involves large remote datasets (e.g., Parquet files on S3), use DuckDB to query the data directly without downloading the entire dataset. Install and load DuckDB extensions (httpfs, parquet) as needed. Use S3 anonymous access unless otherwise specified. Return answers in the requested JSON format, and encode plots as base64 data URIs under 100,000 characters if required.\n"
                    "12. The entire code must be executable in a single run without any user input.\n"
                    "13. *IMPORTANT* The entire script (including data download, processing, and output) must finish in under 3 minutes. Optimize query and code for speed. If the dataset is too large, sample or aggregate as needed to stay within the time limit.\n"
                    "14. Suppress all warnings (e.g., using warnings.filterwarnings('ignore')) so that only the required output is printed.\n"
                    "15. Give only the value of answers to the question what are asked for and nothing else.\n"
                    "16. Always assume headless environment: use matplotlib (Agg) backend, save plots to base64 PNG.\n"
                    "17. Before heavy queries or loops, call `_checkpoint()` to respect runtime/memory limits.\n"
                    "18. For tabular data:\n"
                    "       - Small CSV/JSON: use pandas.read_csv/read_json directly.\n"
                    "       - Large/remote (S3, GCS, parquet): use con = _duckdb_connect().\n"
                    "           - Always filter partitions (year, region, etc.) instead of scanning all.\n"
                    "           - Select only needed columns; avoid SELECT *.\n"
                    "           - Use LIMIT or sample() if data is too large.\n"
                    "19. Always sanitize regex expressions (use raw strings like r'\\$' instead of '\\$').\n"
                    "20. Always validate DataFrame column sizes before plotting (x and y must be equal length).\n"
                    "21. If data fetching fails (404, timeout, etc.), handle gracefully with a fallback message instead of crashing.\n"
                    "22. Use try/except around web scraping, DuckDB queries, and plotting to avoid runtime errors.\n"
                    "23. When parsing HTML (pandas.read_html), ensure 'lxml' is installed and importable; if not, install it.\n"
                    "24. Do not produce duplicate outputs; only print or return the final result.\n"
                    "25. Do not include ```python markdown or explanations — only the executable Python code."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()

RUNTIME_PRELUDE = r"""
        import os, sys, time, signal, resource

        # --- Headless plotting ---
        try:
            import matplotlib
            matplotlib.use("Agg")
        except Exception:
            pass

        # --- Time + memory watchdog ---
        START_TIME = time.time()
        MAX_SECS = int(os.getenv("MAX_RUNTIME", "150"))   # configurable
        MAX_MB   = int(os.getenv("MAX_MEMORY_MB", "450")) # kill before Render OOM

        def _time_mem_check():
            if time.time() - START_TIME > MAX_SECS:
                print('{"error":"time_limit_exceeded"}'); sys.exit(0)
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if usage > MAX_MB:
                print('{"error":"memory_limit_exceeded"}'); sys.exit(0)

        # --- Safe DuckDB connector ---
        def _duckdb_connect():
            import duckdb
            con = duckdb.connect()
            con.execute("SET threads TO 4")
            con.execute("SET enable_progress_bar=false")
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                con.execute("INSTALL parquet; LOAD parquet;")
            except Exception:
                pass
            con.execute("SET hive_partitioning=true")
            con.execute("SET s3_region='ap-south-1'")
            con.execute("SET s3_use_ssl=true")
            con.execute("SET s3_access_key_id=''")
            con.execute("SET s3_secret_access_key=''")
            con.execute("SET s3_external_signing=false")
            return con

        # --- Scatter safety patch ---
        def _safe_scatter(plt):
            _orig = plt.scatter
            def _wrap(x, y, **kw):
                import numpy as np
                try:
                    import pandas as pd
                    if hasattr(x, "dropna"): x = x.dropna()
                    if hasattr(y, "dropna"): y = y.dropna()
                except Exception: pass
                x = np.asarray(x); y = np.asarray(y)
                n = min(len(x), len(y))
                if n == 0: raise ValueError("Not enough data to plot")
                return _orig(x[:n], y[:n], **kw)
            plt.scatter = _wrap
            return plt

        # --- Always check after heavy work ---
        def _checkpoint():
            _time_mem_check()

        import warnings
        warnings.filterwarnings("ignore")
"""

def write_code_to_file(code):
    with open("generated_code.py", "w", encoding="utf-8") as f:
        f.write(RUNTIME_PRELUDE + "\n\n")
        f.write(code)


def extract_required_libraries(code):
    pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z_][\w]*)', re.MULTILINE)
    libraries = pattern.findall(code)
    return sorted(set(libraries))

def install_missing_libraries(libraries):
    pip_mapping = {
        'sklearn': 'scikit-learn',
        'bs4': 'beautifulsoup4',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'requests': 'requests',
        'scipy': 'scipy',
        'plotly': 'plotly',
        'statsmodels': 'statsmodels',
        'sqlalchemy': 'SQLAlchemy',
        'pyarrow': 'pyarrow',
        'pyodbc': 'pyodbc',
        'pymysql': 'PyMySQL',
        'duckdb': 'duckdb',
        'openai': 'openai',
        'dotenv': 'python-dotenv',
        'io': 'io',
        'datetime': 'datetime',
        'base64': 'base64',
        'sqlite3': 'sqlite3',
        'subprocess': 'subprocess',
        'json': 'json',
        're': 're',
        'sys': 'sys',
        'time': 'time',
        'traceback': 'traceback',
        'matplotlib.pyplot': 'matplotlib',
        'openai.OpenAI': 'openai',
        'duckdb.connect': 'duckdb',
        'openai.chat.completions': 'openai',
        'openai.OpenAI': 'openai',
        'openai.chat.completions.create': 'openai',
        'openai.chat.completions.choices': 'openai',
        'openai.chat.completions.choices[0].message.content': 'openai',
        'openai.chat.completions.choices[0].message': 'openai',
        'openai.chat.completions.choices[0]': 'openai',
        'openai.chat.completions.choices': 'openai',
        'openai.chat.completions.create': 'openai'
    }

    for lib in libraries:
        pip_name = pip_mapping.get(lib, lib)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass
            # print(f"Failed to install {pip_name}")

def run_generated_code():
    try:
        result = subprocess.check_output(
            ["python", "generated_code.py"], 
            stderr=subprocess.STDOUT, 
            text=True, 
            timeout = 180
            )
        return True, result.strip()
    except subprocess.TimeoutExpired:
        return False, "❌ Script timed out after 3 minutes."
    except subprocess.CalledProcessError as e:
        return False, e.output.strip()

def generate_and_execute(question):
    output = ""
    for _ in range(RETRIES):
        try:
            code = ask_llm_to_generate_code(question)
            write_code_to_file(code)

            libs = extract_required_libraries(code)
            install_missing_libraries(libs)

            success, output = run_generated_code()
            if success:
                return output
        except Exception:
            continue
    return f"❌ Script error:\n{output}"

@app.post("/ask")
async def ask_question(
    questions_file: UploadFile = File(..., description="The questions.txt file (required)"),
    image: Optional[UploadFile] = File(None, description="Optional image file"),
    dataset: Optional[UploadFile] = File(None, description="Optional dataset file"),
):
    questions_path = "questions.txt"
    with open(questions_path, "wb") as f:
        content = await questions_file.read()
        f.write(content)
    with open(questions_path, "r", encoding="utf-8") as f:
        question = f.read().strip()
    if not question:
        raise HTTPException(status_code=400, detail="questions.txt is empty.")

    # Save image if provided
    if image:
        image_path = image.filename
        with open(image_path, "wb") as f:
            f.write(await image.read())

    # Save dataset if provided
    if dataset:
        dataset_path = dataset.filename
        with open(dataset_path, "wb") as f:
            f.write(await dataset.read())

    output = generate_and_execute(question)
    if output.startswith("❌"):
        raise HTTPException(status_code=500, detail=output)
    return {"result": output}


# -----------if you want to run this script directly, uncomment the following lines-----------
"""
if __name__ == "__main__":
    question = '''
            <add your question here>
            '''
    output = generate_and_execute(question)
"""

"""
working curl request

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: multipart/form-data" -F "questions_file=@questions.txt"

curl -X POST "https://dataanalyst-3dlf.onrender.com/ask" -H "Content-Type: multipart/form-data" -F "questions_file=@questions.txt"
"""