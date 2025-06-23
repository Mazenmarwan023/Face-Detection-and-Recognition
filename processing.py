from pathlib import Path
import random
import shutil
import cv2
from typing import List, Sequence, Set
import re


class FaceDatasetPreprocessor:
    """
    Processes face images from the DataSet directory.
    Images are organized as XX-YY.jpg where XX is person ID and YY is image number.

    Final tree
    └── Processed/
        ├── train/
        │   ├── color/
        │   │   └── person_name/…
        │   └── grayscale/
        │       └── person_name/…
        └── test/
            ├── color/
            │   └── person_name/…
            └── grayscale/
                └── person_name/…
    """

    # List of random names to use
    NAMES = [
        "John", "James", "Michael", "David", "Robert",
        "William", "Thomas", "Daniel", "Paul", "Mark",
        "Donald", "George", "Kenneth", "Steven", "Edward",
        "Brian", "Ronald", "Anthony", "Kevin", "Jason"
    ]

    def __init__(
            self,
            root_dir: str | Path,
            clean_root: str | Path,
            subjects_per_set: int = 9,  # 42-50
            images_per_subject: int = 14,
            train_images: int = 10,
            random_state: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.clean_root = Path(clean_root)
        self.subjects_per_set = subjects_per_set
        self.images_per_subject = images_per_subject
        self.train_images = train_images
        self.random_state = random_state
        random.seed(random_state)
        self._used_names: Set[str] = set()

    def reduce_dataset(self) -> None:
        """Process first set of subjects in color."""
        self._reset_clean_root()
        subjects = self._pick_subjects()
        random_names = self._get_random_names(len(subjects))
        
        for person_id, person_name in zip(subjects, random_names):
            imgs = self._pick_images(person_id)
            train, test = imgs[:self.train_images], imgs[self.train_images:]

            self._copy_images(
                train,
                self.clean_root / "train" / "color" / person_name,
            )
            self._copy_images(
                test,
                self.clean_root / "test" / "color" / person_name,
            )

    def process_grayscale(self) -> None:
        """Process another set of subjects in grayscale."""
        # Get all available subjects
        all_subjects = self._get_all_subjects()
        # Get subjects that weren't used in color processing
        used_subjects = set(self._pick_subjects())
        available_subjects = [s for s in all_subjects if s not in used_subjects]
        
        if len(available_subjects) < self.subjects_per_set:
            raise ValueError(f"Not enough unused subjects for grayscale processing. Need {self.subjects_per_set}, found {len(available_subjects)}")
        
        # Select random subjects from available ones
        subjects = random.sample(available_subjects, self.subjects_per_set)
        random_names = self._get_random_names(len(subjects))
        
        for person_id, person_name in zip(subjects, random_names):
            imgs = self._pick_images(person_id)
            train, test = imgs[:self.train_images], imgs[self.train_images:]

            self._copy_grayscale_images(
                train,
                self.clean_root / "train" / "grayscale" / person_name,
            )
            self._copy_grayscale_images(
                test,
                self.clean_root / "test" / "grayscale" / person_name,
            )

    # alias
    __call__ = reduce_dataset

    def _reset_clean_root(self):
        if self.clean_root.exists():
            shutil.rmtree(self.clean_root)
        for split in ["train", "test"]:
            for mode in ["color", "grayscale"]:
                (self.clean_root / split / mode).mkdir(parents=True, exist_ok=True)
        self._used_names.clear()

    def _get_all_subjects(self) -> List[str]:
        """Get all available subject IDs from the dataset."""
        pattern = re.compile(r'(\d+)-')
        person_ids = set()
        for file in self.root_dir.glob('*.jpg'):
            match = pattern.match(file.name)
            if match:
                person_ids.add(match.group(1))
        return sorted(list(person_ids))

    def _pick_subjects(self) -> List[str]:
        """Pick random subjects from all available subjects."""
        all_subjects = self._get_all_subjects()
        if len(all_subjects) < self.subjects_per_set:
            raise ValueError(f"Need {self.subjects_per_set} subjects, found {len(all_subjects)}")
        return random.sample(all_subjects, self.subjects_per_set)

    def _get_random_names(self, count: int) -> List[str]:
        """Get random names that haven't been used before."""
        available_names = [name for name in self.NAMES if name not in self._used_names]
        if len(available_names) < count:
            raise ValueError(f"Not enough unused names. Need {count}, found {len(available_names)}")
        selected_names = random.sample(available_names, count)
        self._used_names.update(selected_names)
        return selected_names

    def _pick_images(self, person_id: str) -> List[Path]:
        imgs = sorted(
            self.root_dir.glob(f"{person_id}-*.jpg")
        )
        if len(imgs) < self.images_per_subject:
            raise ValueError(f"Person {person_id} has only {len(imgs)} images.")
        return imgs[:self.images_per_subject]

    def _copy_images(self, paths: Sequence[Path], dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, dst_dir / src.name)

    def _copy_grayscale_images(self, paths: Sequence[Path], dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            # Read image and convert to grayscale
            img = cv2.imread(str(src))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Save as grayscale
            cv2.imwrite(str(dst_dir / src.name), gray)


# ──────────────────────────────── CLI demo ───────────────────────────── #
if __name__ == "__main__":
    pre = FaceDatasetPreprocessor(
        root_dir="DataSet",
        clean_root="Processed",
        subjects_per_set=9,
        images_per_subject=14,
        train_images=10,
        random_state=2025,
    )
    # Process first set in color
    pre()
    # Process second set in grayscale
    pre.process_grayscale()
    print("✓ Done. New structure: Processed/{train,test}/{color,grayscale}/person_name/…")