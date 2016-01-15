"""
Have some easy-to-use catalog of globular and open clusters in the same format.
"""

import numpy as np
from astropy.table import Table


def separate_harris_catalog(contents):
    """
    Separate the line-by-line contents of the Harris catalog into the three
    prescribed parts.
    """

    in_part, line_indices = False, []
    for i, line in enumerate(contents):
        if len(line_indices) > 0 and line_indices[-1] == i + 1:
            print("skipping")
            # The line after the 'ID' header is always blank, but we want the
            # rest of the text that is actually in the part.
            continue


        if line.lstrip().startswith("ID"):
            line_indices.append(i + 2)
            in_part = True

        elif in_part and len(line.strip()) == 0:
            line_indices.append(i)
            in_part = False

    # line_indices contains start, end indices for each part in a flat list.
    # Now send back separate parts.
    N = int(len(line_indices)/2) # Should always be 3 parts, but whatever.
    parts = []
    for i in range(N):
        si, ei = line_indices[2*i:2*i+2]
        parts.append(contents[si:ei])

    # Send back the three parts.
    return parts



def parse_machine_readable(line, char_indices):
    """
    Because fuck fortran.
    """

    data_columns = []
    for si, ei, kind in char_indices:
        try:
            _ = kind(line[si:ei].strip())
        except:
            _ = np.nan
        data_columns.append(_)
    return data_columns


def parse_line_part1(line):
    """
    Parse a line from Part I of the Harris catalog.
    """

    columns = ("Name", "RA", "DEC", "l", "b", "R_Sun", "R_gc", "X", "Y", "Z")

    # Get the ID and name first.
    cluster_id, cluster_name = (line[:12].strip(), line[12:25])

    _ = line[25:].split()
    ra, dec = ":".join(_[:3]), ":".join(_[3:6])
    data_columns = [cluster_name.strip(), ra, dec] + list(map(float, _[6:]))

    assert len(columns) == len(data_columns)
    data = dict(zip(columns, data_columns))
    return (cluster_id, data)


def parse_line_part2(line):
    """
    Parse a line from Part II of the Harris catalog.
    """

    columns = ("[Fe/H]", "wt", "E(B-V)", "V_HB", "(m-M)V", "V_t", "M_V,t",
        "U-B", "B-V", "V-R", "V-I", "spt", "ellip")

    # Get the ID first.
    cluster_id = line[:13].strip()

    char_indices = [
        (13, 18, float),
        (19, 21, int),
        (24, 28, float),
        (29, 34, float),
        (35, 40, float),
        (41, 46, float),
        (47, 53, float),
        (55, 60, float),
        (61, 66, float),
        (67, 72, float),
        (73, 78, float),
        (79, 85, str), 
        (86, 90, float)
    ]
    data_columns = parse_machine_readable(line, char_indices)

    assert len(columns) == len(data_columns)

    data = dict(zip(columns, data_columns))
    return (cluster_id, data)



def parse_line_part3(line):
    """
    Parse a line from Part III of the Harris catalog.
    """

    columns = ("V_HELIO", "V_HELIO_ERR", "V_LSR", "sigma_v", "sigma_v_err",
        "c", "r_c", "r_h", "mu_V", "rho_0", "log(tc)", "log(th)")

    # Get the ID first.
    cluster_id = line[:13].strip()
    
    char_indices = [
        (12, 18, float),
        (20, 24, float),
        (25, 32, float),
        (36, 40, float),
        (43, 46, float),
        (49, 53, float),
        (59, 63, float),
        (65, 69, float),
        (72, 77, float),
        (79, 84, float),
        (86, 91, float),
        (92, None, float)
    ]
    data_columns = parse_machine_readable(line, char_indices)
    
    # Create a dictionary.
    assert len(columns) == len(data_columns)
    data = dict(zip(columns, data_columns))
    return (cluster_id, data)



def parse_harris_catalog(filename):
    """
    Read the Harris catalog of globular clusters and merge it into one giant
    table.
    """

    with open(filename, "r") as fp:
        contents = fp.readlines()

    # Parts start with '   ID' and end with blank lines.
    parts = separate_harris_catalog(contents)

    part1 = list(map(parse_line_part1, parts[0]))
    part2 = list(map(parse_line_part2, parts[1]))
    part3 = list(map(parse_line_part3, parts[2]))

    joined_parts = []
    for i, (cluster_id, part1_data) in enumerate(part1):
        
        assert cluster_id == part2[i][0]
        assert cluster_id == part3[i][0]

        joined_data = { "ID": cluster_id }
        joined_data.update(part1_data)
        joined_data.update(part2[i][1])
        joined_data.update(part3[i][1])

        joined_parts.append(joined_data)

    t = Table(rows=joined_parts)

    return t


if __name__ == "__main__":

    catalog = parse_harris_catalog("harris_mwgc.txt")
    catalog.write("harris_mwgc.fits")

