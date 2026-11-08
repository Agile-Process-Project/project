import pandas as pd
import numpy as np

# 1. Load the original dataset
df = pd.read_csv("listings_cleaned.csv")

# 2. Select the columns we need
properties = df[[
    "id",
    "name",
    "street",
    "city",
    "neighbourhood",
    "property_type",
    "room_type",
    "bedrooms",
    "accommodates",
    "price",
    "availability_365"
]].copy()


# 3. Remove listings where the nightly price is zero
price_check = (
    df["price"]
    .astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

price_check = pd.to_numeric(price_check, errors="coerce")

# Keep only rows where price is greater than zero
valid_price = price_check > 0

properties = properties.loc[valid_price].copy()

# Keep the original dataframe in sync for the later price calculation
df = df.loc[valid_price].copy()

print("Listings removed due to zero, missing or invalid price:",
      (~valid_price).sum())


# 4. Rename columns
properties.rename(columns={
    "id": "source_listing_id",
    "name": "property_name",
    "street": "address",
    "city": "locality",
    "accommodates": "max_occupants",
    "availability_365": "availability_days"
}, inplace=True)



# 5. Clean text columns
text_columns = [
    "property_name",
    "address",
    "locality",
    "neighbourhood",
    "property_type",
    "room_type"
]

for column in text_columns:
    properties[column] = (
        properties[column]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

# 5.1 Fill missing locality using the address
missing_locality = properties["locality"].isna()

address_first_part = (
    properties["address"]
    .str.split(",", n=1)
    .str[0]
    .str.strip()
)

# Change "County Dublin" to "Dublin"
address_first_part = address_first_part.replace(
    r"(?i)^County Dublin$", "Dublin", regex=True
)

properties.loc[missing_locality, "locality"] = (
    address_first_part[missing_locality]
)

print(
    "Localities filled:",
    (missing_locality & properties["locality"].notna()).sum()
)

# 5.2 Fill missing neighbourhoods using locality
missing_neighbourhood = properties["neighbourhood"].isna()

properties.loc[missing_neighbourhood, "neighbourhood"] = (
    properties.loc[missing_neighbourhood, "locality"]
)

print(
    "Neighbourhoods filled:",
    (missing_neighbourhood & properties["neighbourhood"].notna()).sum()
)


# 5.3 Clean bedroom values based on room type

# Convert bedrooms to numeric temporarily
properties["bedrooms"] = pd.to_numeric(
    properties["bedrooms"], errors="coerce"
)

# Identify private rooms
private_room_mask = (
    properties["room_type"].str.casefold() == "private room"
)

# Private rooms: set bedrooms to 1 if empty or 0
properties.loc[
    private_room_mask & (
        properties["bedrooms"].isna() |
        (properties["bedrooms"] == 0)
    ),
    "bedrooms"
] = 1

# All other room types: set bedrooms to 0
properties.loc[
    ~private_room_mask,
    "bedrooms"
] = 0

print("Bedroom values updated")


# 5.4 Generate property names where missing

missing_name = properties["property_name"].isna()

properties.loc[missing_name, "property_name"] = (
    properties.loc[missing_name, "bedrooms"].fillna(0)
    .astype(int).astype(str)
    + " Bedroom "
    + properties.loc[missing_name, "property_type"].fillna("Property")
    + " in "
    + properties.loc[missing_name, "neighbourhood"].fillna("Unknown")
    + ", "
    + properties.loc[missing_name, "locality"].fillna("Unknown")
)

print("Property names generated:", missing_name.sum())





# 6. Convert numeric columns
properties["source_listing_id"] = pd.to_numeric(
    properties["source_listing_id"], errors="coerce"
).astype("Int64")

properties["bedrooms"] = pd.to_numeric(
    properties["bedrooms"], errors="coerce"
)

properties["max_occupants"] = pd.to_numeric(
    properties["max_occupants"], errors="coerce"
)

properties["availability_days"] = pd.to_numeric(
    properties["availability_days"], errors="coerce"
)


# 7. Clean nightly price
nightly_price = (
    df["price"]
    .astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

nightly_price = pd.to_numeric(
    nightly_price, errors="coerce"
)

# Reject zero or negative prices
nightly_price = nightly_price.where(nightly_price > 0)


# 8. Calculate estimated monthly rent in cents
properties["monthly_rent_cents"] = (
    nightly_price * 30 * 100
).round().astype("Int64")


# 9. Display the cleaned data
print("\nCLEANED DATA:")
print(properties.head())

print("\nDATASET DIMENSIONS:")
print(properties.shape)

print("\nMISSING VALUES:")
print(properties.isnull().sum())

print("\nDATA TYPES:")
print(properties.dtypes)


"""
Checks
"""

# Check for duplicate source listing IDs
print("\nDUPLICATE SOURCE LISTING IDs:")
print(properties["source_listing_id"].duplicated().sum())

# Check for missing important fields
print("\nMISSING PROPERTY NAMES:")
print(properties["property_name"].isna().sum())

print("\nMISSING ADDRESSES:")
print(properties["address"].isna().sum())

print("\nMISSING BEDROOM COUNTS:")
print(properties["bedrooms"].isna().sum())

print("\nMISSING MONTHLY RENT:")
print(properties["monthly_rent_cents"].isna().sum())

# Check monthly rent calculations
print("\nMONTHLY RENT EXAMPLES:")
print(properties[[
    "source_listing_id",
    "monthly_rent_cents"
]].head(10))


# Save intermediate cleaned dataset
properties.to_csv(
    "properties_prepared.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")

