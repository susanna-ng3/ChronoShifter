import numpy as np
import subprocess
import tempfile
import textwrap
from pathlib import Path


def _build_htdp_update_xyz_script(
    t1,
    t2,
    frame_key_input,
    frame_key_output,
    input_file,
    output_file,
    input_option,
    main_menu_option=4,
):
    """
    Build stdin script for HTDP 3.6.0 'Transform positions between reference
    frames and/or dates' (main menu option 4), using an ASCII XYZ file.

    Sequence (matches your transcript):

      4                       -> main_menu_option
      output_file.name        -> file to contain transformed positions
      frame_key_input               -> reference frame of input positions
      frame_key_output               -> reference frame for output positions
      2                       -> input time as decimal year
      t1
      2                       -> output time as decimal year
      t2
      input_option            -> '4' for X,Y,Z,TEXT file
      input_file.name         -> input XYZ file
      0                       -> return to main menu / exit
    """
    return textwrap.dedent(f"""\
        {main_menu_option}
        {output_file.name}
        {frame_key_input}
        {frame_key_output}
        2
        {t1:.6f}
        2
        {t2:.6f}
        {input_option}
        {input_file.name}
        0
    """)


def htdp_xyz_update_itrf2020(
    X,
    Y,
    Z,
    t1,
    t2,
    htdp_exe="htdp",   # e.g. r"C:\path\to\htdp.exe"
    frame_key_input = 1,       # 1= NAD83(2011)
    frame_key_output=26,      # 26 = ITRF2020 (=IGS20=IGb20=WGS84(G2296))
    input_option=4,    # from your transcript: 4 = X,Y,Z,TEXT file
    main_menu_option=4 # from your transcript: 4 = transform positions
):
    """
    Use HTDP 3.6.0 (mode 4) with XYZ input and return updated XYZ at epoch t2.

    Parameters
    ----------
    X, Y, Z : array-like
        ECEF coordinates in meters at epoch t1 (in the selected HTDP frame).
    t1, t2 : float
        Source and target epochs in decimal years (e.g., 2010.0, 2020.0).
    htdp_exe : str
        Path or name of the HTDP executable.
    frame_key : int
        HTDP reference frame key (26 for ITRF2020 / IGS20 / IGb20 / WGS84(G2296)).
    input_option : int
        HTDP menu option number for "file of X,Y,Z,TEXT" (4 in your transcript).
    main_menu_option : int
        HTDP main menu option number for "Transform positions..." (4).

    Returns
    -------
    X2, Y2, Z2 : numpy.ndarray
        Updated ECEF coordinates at epoch t2, same shape as X/Y/Z.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, Z must all have the same shape")

    if t1 == t2:
        raise ValueError("t1 and t2 must differ")

    n = X.size
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        in_file = tmpdir / "htdp_xyz_input.txt"
        out_file = tmpdir / "htdp_xyz_output.txt"

        # Simple station names: P0, P1, ...
        names = [f"P{i}" for i in range(n)]

        # 1) Write input as: X Y Z TEXT (space-separated)
        with in_file.open("w", newline="\n") as f:
            for x, y, z, name in zip(X_flat, Y_flat, Z_flat, names):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {name}\n")

        # 2) Build the exact command script matching your interactive session
        cmd_script = _build_htdp_update_xyz_script(
            t1=t1,
            t2=t2,
            frame_key_input=frame_key_input,
            frame_key_output=frame_key_output,
            input_file=in_file,
            output_file=out_file,
            input_option=input_option,
            main_menu_option=main_menu_option,
        )

        # 3) Run HTDP
        result = subprocess.run(
            [htdp_exe],
            input=cmd_script,
            text=True,
            cwd=tmpdir,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"HTDP failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        if not out_file.exists():
            raise RuntimeError(
                f"HTDP did not create expected output file {out_file}\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        # 4) Parse updated XYZ in same format: X Y Z TEXT
        X2 = np.zeros(n, dtype=float)
        Y2 = np.zeros(n, dtype=float)
        Z2 = np.zeros(n, dtype=float)

        with out_file.open("r") as f:
            lines = f.readlines()

        if len(lines) < n:
            raise RuntimeError(
                f"Expected at least {n} lines in HTDP output, found {len(lines)}.\n"
                f"File contents:\n{''.join(lines)}"
            )

        name_to_index = {nm: i for i, nm in enumerate(names)}

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x2_str, y2_str, z2_str, nm = parts[0], parts[1], parts[2], parts[3]
            if nm not in name_to_index:
                continue
            idx = name_to_index[nm]
            X2[idx] = float(x2_str)
            Y2[idx] = float(y2_str)
            Z2[idx] = float(z2_str)

    # 5) Reshape back to original
    X2 = X2.reshape(X.shape)
    Y2 = Y2.reshape(Y.shape)
    Z2 = Z2.reshape(Z.shape)

    return X2, Y2, Z2
