from fraud_detection.pipelines.run_eda_pipeline import run


if __name__ == "__main__":
    run(sample=True, nrows=100000)
