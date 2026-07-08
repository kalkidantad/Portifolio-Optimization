"""Download and save YFinance data for TSLA, BND, and SPY."""

from pathlib import Path

from src.data_loader import DEFAULT_END, DEFAULT_START, fetch_all_assets, save_processed_data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"

    print(f"Fetching data from {DEFAULT_START} to {DEFAULT_END}...")
    asset_data = fetch_all_assets(start=DEFAULT_START, end=DEFAULT_END)
    save_processed_data(asset_data, output_dir)

    for ticker, df in asset_data.items():
        print(f"{ticker}: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    main()
