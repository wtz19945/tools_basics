import random
from lxml import etree

def generate_stairs_xml(
    filename="random_stairs.xml",
    num_stairs=50,
    stair_size_range=((0.1, 0.3), (0.2, 0.5), (0.01, 0.05)),
    stair_pos_range=(-1.0, 1.0, -1.0, 1.0),  # x_min, x_max, y_min, y_max
    robot_radius=0.3,  # 20cm footprint radius
    clear_zone_margin=0.05  # extra margin in meters
):
    x_min, x_max, y_min, y_max = stair_pos_range
    max_stair_size_x = stair_size_range[0][1]
    max_stair_size_y = stair_size_range[1][1]

    clear_half_x = robot_radius + max_stair_size_x / 2 + clear_zone_margin
    clear_half_y = robot_radius + max_stair_size_y / 2 + clear_zone_margin

    mj = etree.Element("mujoco", model="stairs_scene")

    # Asset section (optional)
    asset = etree.SubElement(mj, "asset")

    # Worldbody
    worldbody = etree.SubElement(mj, "worldbody")

    stairs_added = 0
    attempts = 0
    max_attempts = num_stairs * 10

    while stairs_added < num_stairs and attempts < max_attempts:
        size_x = random.uniform(*stair_size_range[0])
        size_y = random.uniform(*stair_size_range[1])
        size_z = random.uniform(*stair_size_range[2])

        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = size_z / 2  # so the box rests on the groundplane z=0

        # Check if inside clear zone (expanded by half obstacle size + robot radius)
        if (-clear_half_x <= x <= clear_half_x) and (-clear_half_y <= y <= clear_half_y):
            attempts += 1
            continue

        body = etree.SubElement(worldbody, "body", name=f"stair_{stairs_added}", pos=f"{x} {y} {z}")
        etree.SubElement(body, "geom",
                         type="box",
                         size=f"{size_x/2:.3f} {size_y/2:.3f} {size_z/2:.3f}",
                         rgba="0.5 0.4 0.3 1",
                         contype="1",
                         conaffinity="1",
                         friction="1 0.005 0.0001")
        stairs_added += 1

    tree = etree.ElementTree(mj)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Generated {stairs_added} stairs with clear zone around origin in {filename}")

if __name__ == "__main__":
    generate_stairs_xml()
