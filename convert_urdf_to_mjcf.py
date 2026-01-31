"""
Convert Themis URDF to MuJoCo XML (MJCF) format.

This script converts the URDF file to MuJoCo's native XML format,
which provides better compatibility and performance.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import numpy as np


def euler_to_quat(rpy):
    """Convert roll-pitch-yaw euler angles to quaternion [w, x, y, z]."""
    roll, pitch, yaw = rpy
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]


def convert_urdf_to_mjcf(urdf_path: Path, output_path: Path):
    """
    Convert URDF to basic MJCF format.
    
    Args:
        urdf_path: Path to input URDF file
        output_path: Path to output MJCF file
    """
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Get meshes directory
    meshes_dir = (urdf_path.parent.parent / "meshes").resolve()
    
    # Create MJCF root
    mujoco = ET.Element('mujoco', model=root.get('name', 'robot'))
    
    # Add compiler settings
    compiler = ET.SubElement(mujoco, 'compiler')
    compiler.set('angle', 'radian')
    compiler.set('meshdir', str(meshes_dir))
    compiler.set('eulerseq', 'xyz')
    
    # Add options
    option = ET.SubElement(mujoco, 'option')
    option.set('timestep', '0.002')
    option.set('gravity', '0 0 -9.81')
    
    # Add assets (meshes)
    asset = ET.SubElement(mujoco, 'asset')
    mesh_files = set()
    
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename', '')
        if 'package://TH02-A7/meshes/' in filename:
            mesh_file = filename.split('/')[-1]
            mesh_files.add(mesh_file)
    
    for mesh_file in sorted(mesh_files):
        mesh_elem = ET.SubElement(asset, 'mesh')
        mesh_elem.set('name', mesh_file.replace('.STL', '').replace('.stl', ''))
        mesh_elem.set('file', mesh_file)
    
    # Add worldbody
    worldbody = ET.SubElement(mujoco, 'worldbody')
    
    # Add light
    light = ET.SubElement(worldbody, 'light')
    light.set('diffuse', '.5 .5 .5')
    light.set('pos', '0 0 3')
    light.set('dir', '0 0 -1')
    
    # Add ground plane
    geom = ET.SubElement(worldbody, 'geom')
    geom.set('type', 'plane')
    geom.set('size', '10 10 0.1')
    geom.set('rgba', '.9 .9 .9 1')
    
    # Convert links and joints
    link_map = {}
    joint_info = []
    
    # First pass: collect all links
    for link in root.findall('link'):
        link_name = link.get('name')
        link_map[link_name] = link
    
    # Second pass: collect joint information
    for joint in root.findall('joint'):
        joint_info.append({
            'name': joint.get('name'),
            'type': joint.get('type'),
            'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
            'origin': joint.find('origin'),
            'axis': joint.find('axis'),
            'limit': joint.find('limit')
        })
    
    # Find root link (link with no parent joint)
    child_links = {j['child'] for j in joint_info}
    parent_links = {j['parent'] for j in joint_info}
    root_links = parent_links - child_links
    
    if not root_links:
        # If no clear root, use first link
        root_links = {list(link_map.keys())[0]}
    
    # Create body hierarchy
    def add_body(parent_elem, link_name, parent_joint=None):
        """Recursively add bodies to the MJCF."""
        link = link_map.get(link_name)
        if not link:
            return
        
        # Create body element
        body = ET.SubElement(parent_elem, 'body')
        body.set('name', link_name)
        
        # Add position and orientation from joint origin
        if parent_joint:
            origin = parent_joint.get('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                body.set('pos', xyz)
                
                # Convert RPY to quaternion for more accurate representation
                if rpy != '0 0 0':
                    rpy_values = [float(x) for x in rpy.split()]
                    if any(abs(v) > 1e-6 for v in rpy_values):
                        quat = euler_to_quat(rpy_values)
                        # MuJoCo quaternion format is [w, x, y, z]
                        body.set('quat', f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}')
        
        # Add joint if this is not the root
        if parent_joint:
            joint_elem = ET.SubElement(body, 'joint')
            joint_elem.set('name', parent_joint['name'])
            
            jtype = parent_joint['type']
            if jtype == 'revolute' or jtype == 'continuous':
                joint_elem.set('type', 'hinge')
            elif jtype == 'prismatic':
                joint_elem.set('type', 'slide')
            elif jtype == 'fixed':
                # Fixed joints don't need a joint element in MuJoCo
                body.remove(joint_elem)
                joint_elem = None
            
            # Add axis
            if joint_elem is not None:
                axis = parent_joint.get('axis')
                if axis is not None:
                    axis_xyz = axis.get('xyz', '1 0 0')
                    joint_elem.set('axis', axis_xyz)
                    
                    # Add limits - use reasonable defaults if URDF has zero range
                    limit = parent_joint.get('limit')
                    if limit is not None and jtype != 'continuous':
                        lower = float(limit.get('lower', '-3.14'))
                        upper = float(limit.get('upper', '3.14'))
                        
                        # If URDF has zero range, use reasonable defaults based on joint type
                        if abs(upper - lower) < 0.001:
                            # Set reasonable ranges for humanoid joints
                            lower = -3.14
                            upper = 3.14
                        
                        joint_elem.set('range', f'{lower} {upper}')
        
        # Add inertial properties
        inertial = link.find('inertial')
        if inertial is not None:
            inertial_elem = ET.SubElement(body, 'inertial')
            
            # Position
            origin = inertial.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                inertial_elem.set('pos', xyz)
            
            # Mass
            mass = inertial.find('mass')
            if mass is not None:
                mass_value = max(float(mass.get('value', '1')), 0.001)  # Minimum mass for MuJoCo
                inertial_elem.set('mass', str(mass_value))
            
            # Inertia (convert from URDF format to MuJoCo format)
            inertia = inertial.find('inertia')
            if inertia is not None:
                ixx = float(inertia.get('ixx', '0'))
                iyy = float(inertia.get('iyy', '0'))
                izz = float(inertia.get('izz', '0'))
                ixy = float(inertia.get('ixy', '0'))
                ixz = float(inertia.get('ixz', '0'))
                iyz = float(inertia.get('iyz', '0'))
                
                # Check if off-diagonal elements are significant
                off_diag_threshold = 1e-8
                has_off_diagonal = (abs(ixy) > off_diag_threshold or 
                                  abs(ixz) > off_diag_threshold or 
                                  abs(iyz) > off_diag_threshold)
                
                # Ensure minimum inertia values
                min_inertia = 1e-6
                ixx = max(abs(ixx), min_inertia)
                iyy = max(abs(iyy), min_inertia)
                izz = max(abs(izz), min_inertia)
                
                if has_off_diagonal:
                    # Use full inertia matrix
                    inertial_elem.set('fullinertia', f'{ixx} {iyy} {izz} {ixy} {ixz} {iyz}')
                else:
                    # Use diagonal only
                    inertial_elem.set('diaginertia', f'{ixx} {iyy} {izz}')
        
        # Add visual geometry
        for visual in link.findall('visual'):
            geom = ET.SubElement(body, 'geom')
            geom.set('type', 'mesh')
            
            # Add visual origin transform
            v_origin = visual.find('origin')
            if v_origin is not None:
                xyz = v_origin.get('xyz', '0 0 0')
                rpy = v_origin.get('rpy', '0 0 0')
                if xyz != '0 0 0':
                    geom.set('pos', xyz)
                if rpy != '0 0 0':
                    geom.set('euler', rpy)
            
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    filename = mesh.get('filename', '')
                    if 'package://TH02-A7/meshes/' in filename:
                        mesh_file = filename.split('/')[-1]
                        mesh_name = mesh_file.replace('.STL', '').replace('.stl', '')
                        geom.set('mesh', mesh_name)
            
            # Material color
            material = visual.find('material')
            if material is not None:
                color = material.find('color')
                if color is not None:
                    rgba = color.get('rgba', '0.8 0.8 0.8 1')
                    geom.set('rgba', rgba)
            
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('group', '1')
        
        # Find and add child bodies
        for joint in joint_info:
            if joint['parent'] == link_name:
                add_body(body, joint['child'], joint)
        
        # Add contact points for feet
        if link_name in ['FOOT_R', 'FOOT_L']:
            # Add four corner contact points for stable foot contact
            # Foot dimensions based on Themis foot geometry
            # The foot extends forward (along x in foot frame) and down
            foot_forward = 0.12  # 12cm forward from ankle
            foot_back = -0.05    # 5cm back from ankle
            foot_width = 0.06    # 6cm lateral spread (half width)
            foot_down = 0.08     # 8cm down from ankle center
            
            corners = [
                ('heel_in', foot_back, foot_width, foot_down),
                ('heel_out', foot_back, -foot_width, foot_down),
                ('toe_in', foot_forward, foot_width, foot_down),
                ('toe_out', foot_forward, -foot_width, foot_down),
            ]
            
            for corner_name, x, y, z in corners:
                contact_geom = ET.SubElement(body, 'geom')
                contact_geom.set('name', f'{link_name}_{corner_name}')
                contact_geom.set('type', 'sphere')
                contact_geom.set('size', '0.015')  # 1.5cm radius
                contact_geom.set('pos', f'{x} {y} {z}')
                contact_geom.set('rgba', '0.9 0.1 0.1 0.5')  # Red, semi-transparent
                contact_geom.set('contype', '1')
                contact_geom.set('conaffinity', '1')
                contact_geom.set('condim', '3')
                contact_geom.set('friction', '1.0 0.005 0.0001')
    
    # Add root body with freejoint
    for root_link in root_links:
        root_body = ET.SubElement(worldbody, 'body')
        root_body.set('name', root_link)
        root_body.set('pos', '0 0 1')
        
        # Add freejoint for floating base
        freejoint = ET.SubElement(root_body, 'freejoint')
        freejoint.set('name', 'root_joint')
        
        # Add the root link's content
        link = link_map[root_link]
        
        # Add visual geometry for root
        for visual in link.findall('visual'):
            geom = ET.SubElement(root_body, 'geom')
            geom.set('type', 'mesh')
            
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    filename = mesh.get('filename', '')
                    if 'package://TH02-A7/meshes/' in filename:
                        mesh_file = filename.split('/')[-1]
                        mesh_name = mesh_file.replace('.STL', '').replace('.stl', '')
                        geom.set('mesh', mesh_name)
            
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('group', '1')
        
        # Add children
        for joint in joint_info:
            if joint['parent'] == root_link:
                add_body(root_body, joint['child'], joint)
    
    # Write MJCF file
    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"✓ Converted URDF to MJCF: {output_path}")
    print(f"  - Meshes: {len(mesh_files)}")
    print(f"  - Links: {len(link_map)}")
    print(f"  - Joints: {len(joint_info)}")


def main():
    parser = argparse.ArgumentParser(description='Convert URDF to MuJoCo XML (MJCF)')
    parser.add_argument('--urdf', type=str, 
                       default='themis/urdf/TH02-A7.urdf',
                       help='Path to URDF file')
    parser.add_argument('--output', type=str,
                       default='themis/TH02-A7.xml',
                       help='Path to output MJCF file')
    
    args = parser.parse_args()
    
    urdf_path = Path(args.urdf)
    output_path = Path(args.output)
    
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        return
    
    convert_urdf_to_mjcf(urdf_path, output_path)


if __name__ == '__main__':
    main()
