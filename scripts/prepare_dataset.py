from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 5

ROOT_DIR = Path(r"C:\Users\Ensar\Desktop\CXR8")
CSV_PATH = ROOT_DIR / "Data_Entry_2017_v2020.csv"
IMAGE_DIR = ROOT_DIR / "images" / "images"

OUTPUT_DIR = ROOT_DIR / "cxr_drl" / "data_splits"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def encode_labels(label_text: str) -> dict[str, int]:
    findings = set(str(label_text).split("|"))

    return {
        disease: int(disease in findings)
        for disease in DISEASE_LABELS
    }


def verify_required_files() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")


def load_metadata() -> pd.DataFrame:
    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {
        "Image Index",
        "Finding Labels",
        "Patient ID",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing from the CSV: {sorted(missing_columns)}"
        )

    dataframe["Patient ID"] = dataframe["Patient ID"].astype(int)
    dataframe["image_path"] = dataframe["Image Index"].apply(
        lambda filename: str(IMAGE_DIR / filename)
    )

    encoded = dataframe["Finding Labels"].apply(encode_labels)
    encoded_dataframe = pd.DataFrame(encoded.tolist())

    dataframe = pd.concat(
        [
            dataframe[
                [
                    "Image Index",
                    "Finding Labels",
                    "Patient ID",
                    "Patient Age",
                    "Patient Sex",
                    "View Position",
                    "image_path",
                ]
            ],
            encoded_dataframe,
        ],
        axis=1,
    )

    return dataframe


def verify_images(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("\nChecking image files...")

    dataframe["image_exists"] = dataframe["image_path"].apply(
        lambda path: Path(path).is_file()
    )

    missing = dataframe.loc[
        ~dataframe["image_exists"],
        ["Image Index", "image_path"],
    ]

    print(f"CSV records: {len(dataframe):,}")
    print(f"Images found: {dataframe['image_exists'].sum():,}")
    print(f"Missing images: {len(missing):,}")

    if not missing.empty:
        missing.to_csv(
            OUTPUT_DIR / "missing_images.csv",
            index=False,
        )
        raise RuntimeError(
            "Some images referenced in the metadata were not found. "
            "See missing_images.csv."
        )

    return dataframe.drop(columns=["image_exists"])


def create_patient_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = dataframe["Patient ID"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)

    total_patients = len(patients)

    train_end = int(total_patients * TRAIN_RATIO)
    validation_end = train_end + int(total_patients * VALIDATION_RATIO)

    train_patients = set(patients[:train_end])
    validation_patients = set(patients[train_end:validation_end])
    test_patients = set(patients[validation_end:])

    train_dataframe = dataframe[
        dataframe["Patient ID"].isin(train_patients)
    ].copy()

    validation_dataframe = dataframe[
        dataframe["Patient ID"].isin(validation_patients)
    ].copy()

    test_dataframe = dataframe[
        dataframe["Patient ID"].isin(test_patients)
    ].copy()

    return train_dataframe, validation_dataframe, test_dataframe


def check_patient_overlap(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    train_patients = set(train_dataframe["Patient ID"])
    validation_patients = set(validation_dataframe["Patient ID"])
    test_patients = set(test_dataframe["Patient ID"])

    overlaps = {
        "train_validation": train_patients.intersection(validation_patients),
        "train_test": train_patients.intersection(test_patients),
        "validation_test": validation_patients.intersection(test_patients),
    }

    overlap_counts = {
        name: len(values)
        for name, values in overlaps.items()
    }

    print("\nPatient overlap check:")
    for name, count in overlap_counts.items():
        print(f"{name}: {count}")

    if any(overlap_counts.values()):
        raise RuntimeError(
            "Patient leakage detected between dataset subsets."
        )


def save_class_distribution(
    name: str,
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    distribution = {
        disease: int(dataframe[disease].sum())
        for disease in DISEASE_LABELS
    }

    pd.DataFrame(
        {
            "Disease": list(distribution.keys()),
            "Positive Images": list(distribution.values()),
        }
    ).to_csv(
        OUTPUT_DIR / f"{name}_class_distribution.csv",
        index=False,
    )

    return distribution


def print_split_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    print(
        f"{name:<12} "
        f"images={len(dataframe):>7,} | "
        f"patients={dataframe['Patient ID'].nunique():>6,}"
    )


def main() -> None:
    set_seed(SEED)
    verify_required_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_metadata()
    dataframe = verify_images(dataframe)

    train_dataframe, validation_dataframe, test_dataframe = (
        create_patient_split(dataframe)
    )

    check_patient_overlap(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    train_dataframe.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )
    validation_dataframe.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )
    test_dataframe.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("\nDataset split summary:")
    print_split_summary("Train", train_dataframe)
    print_split_summary("Validation", validation_dataframe)
    print_split_summary("Test", test_dataframe)

    distributions = {
        "train": save_class_distribution(
            "train",
            train_dataframe,
        ),
        "validation": save_class_distribution(
            "validation",
            validation_dataframe,
        ),
        "test": save_class_distribution(
            "test",
            test_dataframe,
        ),
    }

    summary = {
        "seed": SEED,
        "dataset": "NIH ChestX-ray14",
        "total_images": len(dataframe),
        "total_patients": int(dataframe["Patient ID"].nunique()),
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "splits": {
            "train": {
                "images": len(train_dataframe),
                "patients": int(
                    train_dataframe["Patient ID"].nunique()
                ),
            },
            "validation": {
                "images": len(validation_dataframe),
                "patients": int(
                    validation_dataframe["Patient ID"].nunique()
                ),
            },
            "test": {
                "images": len(test_dataframe),
                "patients": int(
                    test_dataframe["Patient ID"].nunique()
                ),
            },
        },
        "class_distributions": distributions,
    }

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"\nFiles saved to:\n{OUTPUT_DIR}")
    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
