from re import L, U
import numpy as np

import MDAnalysis as mda

from MDAnalysis.converters.RDKit import atomgroup_to_mol
from MDAnalysis.topology.guessers import guess_types, guess_atom_element

import psi4

from rdkit import Chem


def get_spin_multiplicity(molecule: Chem.Mol) -> int:
    """Returns the spin multiplicity of a :class:`RDKit.Mol`.
    Based on method in http://www.mayachemtools.org/docs/modules/html/code/RDKitUtil.py.html .

    :Arguments:
        *molecule*
            :class:`RDKit.Mol` object
    """
    radical_electrons: int = 0

    for atom in molecule.GetAtoms():
        radical_electrons += atom.GetNumRadicalElectrons()

    total_spin: int = radical_electrons // 2
    spin_mult: int = total_spin + 1
    return spin_mult

def get_new_pos(backbone: mda.AtomGroup, length: float):
    c_pos = backbone.select_atoms('name C').positions[0]
    o_pos = backbone.select_atoms('name O').positions[0]
    a_pos = backbone.select_atoms('name CA').positions[0]
    o_pos = o_pos - c_pos  # Translate coords such that C in at origin
    a_pos = a_pos - c_pos
    o_norm = o_pos/np.linalg.norm(o_pos)
    a_norm = a_pos/np.linalg.norm(a_pos)
    h_pos = -(o_norm + a_norm)
    h_norm = h_pos/np.linalg.norm(h_pos)
    h_norm = (h_norm * length) + c_pos
    return h_norm

def protonate_backbone(resid: mda.AtomGroup, length: float = 1.128) -> mda.Universe:
    backbone = resid.select_atoms('backbone')

    protonated: mda.Universe = mda.Universe.empty(n_atoms=resid.n_atoms + 1, trajectory=True)
    protonated.add_TopologyAttr('masses', [x for x in resid.masses] + [1])
    protonated.add_TopologyAttr('name', [x for x in resid.names] + ['H*'])
    protonated.add_TopologyAttr('types', guess_types(protonated.atoms.names))
    protonated.add_TopologyAttr('elements', [guess_atom_element(atom) for atom in protonated.atoms.names])
    new_pos = resid.positions
    h_pos = get_new_pos(backbone, length)
    protonated.atoms.positions = np.row_stack((new_pos, h_pos))
    return protonated

def opt_geometry(system: mda.Universe, basis: str = 'scf/cc-pvdz') -> float:
    resid: mda.AtomGroup = system.select_atoms('all')
    rd_mol: Chem.Mol = atomgroup_to_mol(resid)
    coords: str = f'{Chem.GetFormalCharge(rd_mol)} {get_spin_multiplicity(rd_mol)}'
    freeze_list: str = ''
    for n in range(len(system.atoms)):
        atom = system.atoms[n]
        coords += f'\n{atom.element} {atom.position[0]} {atom.position[1]} {atom.position[2]}'
        if atom.name != 'H*':
            freeze_list += f'\n{n + 1} xyz'
        else:
            h_ind = n
        if atom.name == 'C':
            c_ind = n
    
    coords += '\nunits angstrom'

    mol: psi4.core.Molecule = psi4.geometry(coords)
    psi4.set_memory('48GB')
    psi4.set_num_threads(12)
    psi4.set_output_file
    psi4.set_options({'reference': 'rhf', 'freeze_core': 'true'})
    psi4.optimize(basis, freeze_list=freeze_list, opt_cooridnates='cartesian',  molecule=mol)
   
    opt_coords = mol.create_psi4_string_from_molecule()
    opt_coords = opt_coords.split('\n')[4:]

    h_line = opt_coords[h_ind].split()
    h_coord = [float(h_line[1]), float(h_line[2]), float(h_line[3])]

    c_line = opt_coords[c_ind].split()
    c_coord = [float(c_line[1]), float(c_line[2]), float(c_line[3])]
    return np.sqrt((c_coord[0] - h_coord[0])**2 + (c_coord[1] - h_coord[1])**2 + (c_coord[2] - h_coord[1])**2)

def list_avg(list):
    tot = 0
    for item in list:
        tot += item
    return item/len(list)

if __name__ == '__main__':

    lengths = {
        'ALA': [],
        'ARG': [],
        'ASN': [],
        'ASP': [],
        'CYS': [],
        'GLU': [],
        'GLN': [],
        'GLY': [],
        'HIS': [],
        'ILE': [],
        'LUE': [],
        'LYS': [],
        'MET': [],
        'PHE': [],
        'PRO': [],
        'SER': [],
        'THR': [],
        'TRP': [],
        'TYR': [],
        'VAL': []
    }

    res_names = [x for x in lengths.keys()]

    with open('results', 'w+') as output:
        for n in res_names:
            U = mda.Universe(f'{n}.pdb')
            step0 = U.select_atoms('all')
            step1 = protonate_backbone(step0)
            length0 = opt_geometry(step1)
            length1 = opt_geometry(step1)
            length2 = opt_geometry(step1)

            lengths[n] += [length0]
            lengths[n] += [length1]
            lengths[n] += [length2]

            output.write(f'{n}, {list_avg(lengths[n])}\n')

    print(0)
