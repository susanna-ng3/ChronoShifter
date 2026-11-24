import os
import graphviz

# --- Add GRAPHVIZ TO PATH (Update this to your local path) ---
# os.environ["PATH"] += os.pathsep + r"C:/Path/To/Graphviz/bin"
os.environ["PATH"] += os.pathsep + r"C:/Users/engsu.ONID/Downloads/windows_10_cmake_Release_Graphviz-14.0.5-win64/Graphviz-14.0.5-win64/bin"

OUT_BASENAME = "chronoshifter_flowchart_optimized_v2"


def build_dot(fmt: str) -> graphviz.Digraph:
    dot = graphviz.Digraph("chronoshifter_flowchart_final", comment="ChronoShifter Workflow", format=fmt)

    # Page layout - size! forces the stretch to fill the 8.5x11 area
    dot.attr(
        rankdir="TD",
        size="8.3,10.8!",
        ratio="fill",
        nodesep="1.2",  # Horizontal gap between columns
        ranksep="0.25",  # Shortened arrows (vertical gap)
        pad="0.1",
    )

    # Global node styling
    dot.attr(
        "node",
        shape="rect",
        style="rounded",
        fontname="Arial Bold",
        fontsize="24",  # Large body text
        margin="0.3,0.2",
        penwidth="2.5",  # Thick borders for visibility
        width="4.5",  # Minimum width to fill horizontal space
    )

    # HTML label helper
    def tbl(lines, title=None, pt_title=34, pt_body=24):
        rows = []
        if title is not None:
            rows.append(f'<TR><TD ALIGN="CENTER"><font point-size="{pt_title}"><b>Step {title}</b></font></TD></TR>')
        for line in lines:
            # Formatting subscripts for coordinates
            line = line.replace("e0", "e<sub>0</sub>").replace("n0", "n<sub>0</sub>").replace("H0", "H<sub>0</sub>")
            line = line.replace("lambda0", "&lambda;<sub>0</sub>").replace("phi0", "&phi;<sub>0</sub>").replace("h0",
                                                                                                                "h<sub>0</sub>")
            line = line.replace("X0", "X<sub>0</sub>").replace("Y0", "Y<sub>0</sub>").replace("Z0", "Z<sub>0</sub>")
            line = line.replace("X1", "X<sub>1</sub>").replace("Y1", "Y<sub>1</sub>").replace("Z1", "Z<sub>1</sub>")
            line = line.replace("lambda1", "&lambda;<sub>1</sub>").replace("phi1", "&phi;<sub>1</sub>").replace("h1",
                                                                                                                "h<sub>1</sub>")
            line = line.replace("e1", "e<sub>1</sub>").replace("n1", "n<sub>1</sub>").replace("H1", "H<sub>1</sub>")

            rows.append(f'<TR><TD ALIGN="CENTER"><font point-size="{pt_body}">{line}</font></TD></TR>')
        return (
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8">'
                + "".join(rows)
                + "</TABLE>>"
        )

    # Nodes
    dot.node("Start", "Input LAS/LAZ", shape="oval", style="filled", fillcolor="#E6E6FA", fontsize="32", width="5.0")

    dot.node("Step1", tbl(["e0, n0, H0", "Source CRS, Geoid18/12B"], title="1"))
    dot.node("Step2", tbl(["lambda0, phi0, H0"], title="2"), width="3.0")
    dot.node("Step3", tbl(["remove Geoid18/12B"], title="3"), width="3.0")

    dot.node("NumPy", tbl(["using numpy array", "h0"]))
    dot.node("GeodCoords", tbl(["lambda0, phi0, h0"]), width="3.0")

    dot.node("Step4", tbl(["using pyproj", "ECEF X0, Y0, Z0", "NAD83(2011) at 2010.0"], title="4"))
    dot.node("Step5", tbl(["14-parameter Helmert transform", "applied to ALL points"], title="5"))

    # Step 6 - Extra wide to remove white space on the right
    dot.node(
        "Step6",
        tbl(
            [
                "HTDP Velocity Correction:",
                "&bull; Select center point  &bull; Run HTDP.exe (&Delta;)  &bull; Apply &Delta; offset to ALL points",
            ],
            title="6",
        ),
        style="filled,rounded",
        fillcolor="#FFE6CC",
        width="7.8",  # Spans nearly the whole page width
    )

    dot.node("Step7", tbl(["ECEF X1, Y1, Z1", "ITRF2020 at 2020.0"], title="7"))
    dot.node("Convert", tbl(["lambda1, phi1, h1"]), width="3.0")
    dot.node("Step8", tbl(["+Geoid22"], title="8"), width="3.0")
    dot.node("Step9", tbl(["apply SPCS22 target zone", "e1, n1, h1"], title="9"))
    dot.node("Step10", tbl(["final coordinates", "e1, n1, H1"], title="10"))

    dot.node("FinalGeod", tbl(["H1"]), width="3.0")

    dot.node("End", "Output LAS/LAZ", shape="oval", style="filled", fillcolor="#E6E6FA", fontsize="32", width="5.0")

    # Edges
    dot.attr("edge", penwidth="2.5", arrowsize="1.5")
    dot.edge("Start", "Step1")
    dot.edge("Step1", "NumPy")
    dot.edge("NumPy", "Step4")
    dot.edge("Step4", "Step5")
    dot.edge("Step5", "Step6")
    dot.edge("Step6", "Step7")
    dot.edge("Step7", "Step9")
    dot.edge("Step9", "Step10")
    dot.edge("Step10", "End")

    dot.edge("Step1", "Step2", constraint="false")
    dot.edge("Step2", "Step3")
    dot.edge("Step3", "GeodCoords")
    dot.edge("GeodCoords", "NumPy", constraint="false")

    dot.edge("Step7", "Convert", constraint="false")
    dot.edge("Convert", "Step8")
    dot.edge("Step8", "FinalGeod")
    dot.edge("FinalGeod", "Step10", constraint="false")

    # Grouping ranks to keep the two-column structure tight
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("Step1");
        s.node("Step2")
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("NumPy");
        s.node("GeodCoords")
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("Step7");
        s.node("Convert")
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("Step9");
        s.node("Step8")
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("Step10");
        s.node("FinalGeod")

    return dot


if __name__ == "__main__":
    build_dot("pdf").render(filename=OUT_BASENAME, cleanup=True)
    build_dot("png").render(filename=OUT_BASENAME, cleanup=True)
    print(f"Generated: {OUT_BASENAME}.pdf and .png")