FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY cleaning/requirements.txt cleaning/requirements.txt
RUN python -m pip install --no-cache-dir -r cleaning/requirements.txt

COPY cleaning/ cleaning/
COPY tests/ tests/

RUN python -m unittest discover -s tests -v

COPY ["Data/Cleaned Data/listings_cleaned.csv", "data/listings_cleaned.csv"]

CMD ["python", "cleaning/clean_listings.py", "--input", "data/listings_cleaned.csv", "--output", "cleaning_output"]