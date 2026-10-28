from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

from steel_energy.config import UCI_DATASET_URL, UCI_DOWNLOAD_PATH


def download_dataset() -> None:
    UCI_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = Request(UCI_DATASET_URL, headers={"User-Agent": "steel-energy-student-project"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        csv_names = [
            name for name in archive.namelist()
            if name.lower().endswith("steel_industry_data.csv")
        ]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one steel dataset CSV in UCI archive; found {csv_names}")
        UCI_DOWNLOAD_PATH.write_bytes(archive.read(csv_names[0]))
    print(f"Downloaded {UCI_DOWNLOAD_PATH}")


if __name__ == "__main__":
    download_dataset()
