# ANN_pipeline/data/pdf.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import time

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KernelDensity

@dataclass
class PdfMapConfig:
    """
    Configuration options for pdfMap.

    bins      : [Nc, NZ] pair
    map_type  : string from {"kde", "hist"}
    pdf_type  : string from {"joint", "marginal"}
    timesteps : Indices into `pdfMap.timename`
    band      : half-width of the filter cube (in cell indices)
    jump      : stride between neighbouring filter centers (in cell indices)
    fil_qty   : if True, store filtered quantities (inputs, rhoBar, wBar)
    """
    bins: Sequence[int]
    map_type: str
    pdf_type: str
    timesteps: Sequence[int]
    band: int
    jump: int
    fil_qty: bool = True

class pdfMap:
    """
    Build input–target datasets for ANN training from 3D DNS data.

    Supported map types:
      - "kde"  : 2D Gaussian KDE in (c, lnZ)
      - "hist" : 2D histogram in (c, lnZ)

    Supported PDF types:
      - "joint"     : joint PDF
      - "marginal"  : marginal PDF
    """

    # Default stoichiometric and flammability limits for H2
    Zst: float = 0.03
    Zl: float = 0.0009
    Zr: float = 0.344

    # Directory names corresponding to timesteps (1-based indexing)
    timename: List[str] = [
        "051", "193", "280", "387", "489",
        "579", "626", "736", "801", "936",
    ]

    def __init__(self, bins, mapID, pdfType, timestep, band, jump, filQty: bool = True):
        cfg = PdfMapConfig(
            bins=bins,
            map_type=mapID,
            pdf_type=pdfType,
            timesteps=list(timestep),
            band=int(band),
            jump=int(jump),
            fil_qty=bool(filQty),
        )
        self.cfg = cfg

        # Flags: pdf calculation method used
        self.kde_map: bool = False
        self.hist_map: bool = False

        # Flags: pdf type calculated
        self.joint_pdf: bool = False
        self.marginal_pdf: bool = False

        # Targets
        self.target_kde: list[np.ndarray] = []
        self.target_hist: list[np.ndarray] = []

        # Inputs / filtered quantities
        self.inputs: list[list[float]] = []
        self.rhoBar: list[float] = []
        self.wBar: list[float] = []

        # Axes (initialized in set_axis)
        self.Nc_kde = self.NZ_kde = None
        self.cSpace_kde = self.ZSpace_kde = self.lnZSpace_kde = None
        self.cCenter_kde = self.cWidth_kde = None
        self.lnZCenter_kde = self.lnZWidth_kde = None
        self.ZCenter_kde = self.ZWidth_kde = None

        self.Nc_hist = self.NZ_hist = None
        self.cSpace_hist = self.ZSpace_hist = self.lnZSpace_hist = None
        self.cCenter_hist = self.cWidth_hist = None
        self.lnZCenter_hist = self.lnZWidth_hist = None
        self.ZCenter_hist = self.ZWidth_hist = None

        # Map requested types to pdf metohd specification
        Nc, NZ = self.cfg.bins
        if self.cfg.map_type == "kde":
            self.kde_map = True
            self.Nc_kde, self.NZ_kde = int(Nc), int(NZ)
        elif self.cfg.map_type == "hist":
            self.hist_map = True
            self.Nc_hist, self.NZ_hist = int(Nc), int(NZ)
        else:
            raise ValueError(f"Unsupported mapID '{m}'. Allowed: 'kde', 'hist'.")

        # Map requested types to pdf type specification
        if cfg.pdf_type == "joint":
            self.joint_pdf = True
        elif cfg.pdf_type == "marginal":
            self.marginal_pdf = True
        else:
            raise ValueError(f"Unsupported pdfType '{m}'. Allowed: 'joint', 'marginal'.")

        if self.joint_pdf and self.marginal_pdf:
            raise ValueError("Please choose only one pdfType: 'joint' or 'marginal'.")

        self.count_ts: int = 0

    # ------------------------------------------------------------------
    # Axis construction
    # ------------------------------------------------------------------
    def _build_Z_space(self, NZ: int) -> np.ndarray:
        """
        Construct a non-uniform Z discretization biased towards flammability
        limits, length NZ in [0, 1]:
        - NZ-6 interior points between Zl and Zr
        - 2 points between 0 and Zl
        - 4 points between Zr and 1
        """
        if NZ < 7:
            raise ValueError("NZ must be at least 7 for the non-uniform Z grid.")
        Z_space = np.linspace(self.Zl, self.Zr, NZ - 6)
        Z_space = np.concatenate((np.linspace(0.0, self.Zl, 3)[:-1], Z_space))
        Z_space = np.concatenate((Z_space, np.linspace(self.Zr, 1.0, 5)[1:]))
        return Z_space

    def set_axis(self) -> None:
        """Initialize discretisation in c and ln Z for all requested map types."""
        # KDE axes
        if self.kde_map:
            self.cSpace_kde = np.linspace(0.0, 1.0, self.Nc_kde)
            self.ZSpace_kde = self._build_Z_space(self.NZ_kde)

            lnZ_edges = np.log(self.ZSpace_kde[1:] / self.Zst)
            lnZ_edges = np.insert(
                lnZ_edges, 0, np.floor(np.log(self.ZSpace_kde[1] / self.Zst))
            )

            self.lnZSpace_kde = lnZ_edges
            self.cCenter_kde = 0.5 * (self.cSpace_kde[:-1] + self.cSpace_kde[1:])
            self.cWidth_kde = self.cSpace_kde[1:] - self.cSpace_kde[:-1]
            self.lnZCenter_kde = 0.5 * (lnZ_edges[:-1] + lnZ_edges[1:])
            self.lnZWidth_kde = lnZ_edges[1:] - lnZ_edges[:-1]
            self.ZCenter_kde = np.exp(self.lnZCenter_kde) * self.Zst
            self.ZWidth_kde = (np.exp(lnZ_edges[1:]) - np.exp(lnZ_edges[:-1])) * self.Zst    

        # Histogram axes
        if self.hist_map:
            self.cSpace_hist = np.linspace(0.0, 1.0, self.Nc_hist)
            self.ZSpace_hist = self._build_Z_space(self.NZ_hist)

            lnZ_edges = np.log(self.ZSpace_hist[1:] / self.Zst)
            lnZ_edges = np.insert(
                lnZ_edges, 0, np.floor(np.log(self.ZSpace_hist[1] / self.Zst))
            )

            self.lnZSpace_hist = lnZ_edges
            self.cCenter_hist = 0.5 * (self.cSpace_hist[:-1] + self.cSpace_hist[1:])
            self.cWidth_hist = self.cSpace_hist[1:] - self.cSpace_hist[:-1]
            self.lnZCenter_hist = 0.5 * (lnZ_edges[:-1] + lnZ_edges[1:])
            self.lnZWidth_hist = lnZ_edges[1:] - lnZ_edges[:-1]
            self.ZCenter_hist = np.exp(self.lnZCenter_hist) * self.Zst
            self.ZWidth_hist = (np.exp(lnZ_edges[1:]) - np.exp(lnZ_edges[:-1])) * self.Zst

    # ------------------------------------------------------------------
    # Core extraction helpers
    # ------------------------------------------------------------------
    def _load_mat(self, base_dir: Path, timestep_idx: int) -> dict:
        """Load a single snapshot .mat file given the timestep index."""
        name = self.timename[timestep_idx - 1]
        mat_path = base_dir / name / f"ts{name}_FS{self.cfg.band * 2 + 1}_Jump{self.cfg.band * 4 + 2}.mat"
        print(f"Reading data at time step: {name} ({mat_path})")
        return loadmat(str(mat_path))

    def _compute_lnZ(self, Z: np.ndarray) -> np.ndarray:
        """Compute ln(Z / Zst) with a lower bound of exp(-5)*Zst and clip to [-5, max]."""
        Z_min = np.exp(-5.0) * self.Zst
        Z = np.maximum(Z, Z_min)
        lnZ = np.log(Z / self.Zst)
        return np.clip(lnZ, -5.0, np.max(lnZ))

    def _iter_filter_boxes(self, shape: Sequence[int]):
        """Generate (i, j, k, a, b, c) for the moving filter box centers."""
        NX, NY, NZ = shape
        band = self.cfg.band
        jump = self.cfg.jump
        a = -1
        for i in range(band, NX - band, jump):
            a += 1
            b = -1
            for j in range(band, NY - band, jump):
                b += 1
                c = -1
                for k in range(band, NZ - band, jump):
                    c += 1
                    yield i, j, k, a, b, c

    def _append_filtered_inputs(self, matData, a: int, b: int, c: int) -> None:
        """Store filtered quantities for ANN input."""
        inputs = [
            float(matData["cTld"][a, b, c]),
            float(matData["gC"][a, b, c]),
            float(np.log(matData["ZTld"][a, b, c] / self.Zst)),
            float(matData["gZ"][a, b, c]),
        ]
        self.inputs.append(inputs)
        self.rhoBar.append(float(matData["rhoBar"][a, b, c]))
        self.wBar.append(float(matData["wBar"][a, b, c]))

    def _kde_pdf(self, cB: np.ndarray, lnZB: np.ndarray) -> np.ndarray:
        """
        Compute a normalized 2D KDE PDF over (c, lnZ) on the configured grid.
        Returns an array of shape (Nc_kde - 1, NZ_kde - 1).
        """
        sample = np.empty((cB.size, 2), dtype=float)
        sample[:, 0] = cB.ravel()
        sample[:, 1] = lnZB.ravel()

        cAxis, lnZAxis = np.meshgrid(self.cCenter_kde, self.lnZCenter_kde, indexing="ij")
        eval_points = np.column_stack((cAxis.ravel(), lnZAxis.ravel()))

        kde = KernelDensity(kernel="epanechnikov", bandwidth=0.1)
        kde.fit(sample)
        log_pdf = kde.score_samples(eval_points)
        pdf = np.exp(log_pdf).reshape(cAxis.shape)

        # Normalize PDF over (c, ln Z)
        area = self.cWidth_kde[:, None] * self.lnZWidth_kde[None, :]
        total = np.sum(pdf * area)
        if total > 0:
            pdf /= total
        return pdf  # (Nc-1, NZ-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_map(self, data_dir: str | Path = ".") -> None:
        """
        Main entry point: loop over timesteps and populate targets / inputs.

        Parameters
        ----------
        data_dir : base directory where the timestep folders (e.g. '489') live.
                   This is '.' for runing from ANN_pipeline/data/.
        """
        data_dir = Path(data_dir)
        self.count_ts = 0
        
        for ts in self.cfg.timesteps:
            matData = self._load_mat(data_dir, ts)
            lnZ = self._compute_lnZ(matData["Z"])
            NX, NY, NZ = lnZ.shape
            nx, ny, nz = matData["wBar"].shape
            dataNum = nx * ny * nz

            print("Generating map ...")
            t0 = time.time()

            count_reacting = 0
            for n_box, (i, j, k, a, b, c) in enumerate(
                self._iter_filter_boxes((NX, NY, NZ)), start=1
            ):
                if n_box % 2000 == 0:
                    t1 = time.time()
                    progress = 100.0 * n_box / dataNum
                    print(f"Progress --> {progress:.1f}%")
                    print(f"Time elapsed: {t1 - t0:.1f} s")

                # Reacting-region filter
                if not (
                    0.02 <= matData["cTld"][a, b, c] <= 0.98
                    and matData["ZTld"][a, b, c] >= 0.0005
                ):
                    continue

                count_reacting += 1

                # Extract local DNS data in the filter cube
                sl_i = slice(i - self.cfg.band, i + self.cfg.band + 1)
                sl_j = slice(j - self.cfg.band, j + self.cfg.band + 1)
                sl_k = slice(k - self.cfg.band, k + self.cfg.band + 1)

                rho_bar = matData["rhoBar"][a, b, c]

                cB = matData["C"][sl_i, sl_j, sl_k] * matData["RHO"][sl_i, sl_j, sl_k] / rho_bar
                lnZB = lnZ[sl_i, sl_j, sl_k] * matData["RHO"][sl_i, sl_j, sl_k] / rho_bar

                # KDE-based target
                if self.kde_map:
                    pdf = self._kde_pdf(cB, lnZB)
                    area = self.cWidth_kde[:, None] * self.lnZWidth_kde[None, :]
                    joint_prob = pdf * area
                    if self.joint_pdf:
                        self.target_kde.append(joint_prob.flatten())
                    elif self.marginal_pdf:
                        prob_c = np.sum(joint_prob, axis=1)
                        prob_lnZ = np.sum(joint_prob, axis=0)
                        self.target_kde.append(np.concatenate([prob_c, prob_lnZ]))

                # Histogram-based target
                if self.hist_map:
                    H, c_edges, lnZ_edges = np.histogram2d(
                        cB.ravel(),
                        lnZB.ravel(),
                        bins=(self.cSpace_hist, self.lnZSpace_hist),
                    )
                    H /= H.sum() if H.sum() > 0 else 1.0
                    if self.joint_pdf:
                        self.target_hist.append(H.flatten())
                    elif self.marginal_pdf:
                        prob_c = H.sum(axis=1)
                        prob_lnZ = H.sum(axis=0)
                        self.target_hist.append(np.concatenate([prob_c, prob_lnZ]))

                if self.cfg.fil_qty:
                    self._append_filtered_inputs(matData, a, b, c)

            self.count_ts += 1
            print(f"Data size collected from timestep {self.timename[ts - 1]} : {count_reacting} | Total: {dataNum}")

        print("Extraction completed.")
        if self.kde_map:
            print(f"Total KDE samples: {len(self.target_kde)}")
        if self.hist_map:
            print(f"Total HIST samples: {len(self.target_hist)}")
        if self.cfg.fil_qty:
            print(f"Total input samples: {len(self.inputs)}")

if __name__ == "__main__":
    # Example usage: single timestep 5 relative to current script directory.
    timeStep = [9]        # 1 - 10: {"051", "193", "280", "387", "489", "579", "626", "736", "801", "936"}
    data_dir = Path(".")

    dataSet = pdfMap(
        bins=[46, 56],
        mapID="hist",
        pdfType="marginal",
        timestep=timeStep,
        band=4,
        jump=18,
        filQty=True,
    )

    time_start = pdfMap.timename[timeStep[0] - 1]
    time_end = pdfMap.timename[timeStep[-1] - 1]

    dataSet.set_axis()
    dataSet.extract_map(data_dir=data_dir)

    print("Saving data ...")
    suffix = f"{dataSet.count_ts}snapshots_from_{time_start}_to_{time_end}_fs_{dataSet.cfg.band * 2 + 1}_jump_{dataSet.cfg.band * 4 + 2}"

    if dataSet.kde_map:
        np.save(f"Target_kde_{suffix}", np.asarray(dataSet.target_kde))
    if dataSet.hist_map:
        np.save(f"Target_hist_{suffix}", np.asarray(dataSet.target_hist))
    if dataSet.cfg.fil_qty:
        np.save(f"Input_{suffix}", np.asarray(dataSet.inputs))
        filData = np.column_stack([dataSet.wBar, dataSet.rhoBar])
        np.save(f"Filtered_data_{suffix}", filData)
