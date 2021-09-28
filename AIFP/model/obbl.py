from __future__ import print_function

from scipy.optimize import fmin_l_bfgs_b

from itertools import chain
from subprocess import check_output
import warnings
from tempfile import NamedTemporaryFile
import logging

import gzip
from base64 import b64encode
from six import PY3, text_type

import numpy as np
from sklearn.utils.deprecation import deprecated
try:
    from openbabel import pybel, openbabel as ob
    from openbabel.pybel import *
    from openbabel.openbabel import OBAtomAtomIter, OBAtomBondIter, OBTypeTable
except ImportError:
    import pybel
    import openbabel as ob
    from pybel import *
    from openbabel import OBAtomAtomIter, OBAtomBondIter, OBTypeTable
#     ob.OBIterWithDepth.__next__ = ob.OBIterWithDepth.next

# from model.toolkits.utils import check_molecule
from model.toolkits.common import detect_secondary_structure, canonize_ring_path
# setup typetable to translate atom types
typetable = OBTypeTable()
typetable.SetFromType('INT')
typetable.SetToType('SYB')


class Molecule(pybel.Molecule):
    def __init__(self, OBMol=None, protein=False):
        # debug
        self.debug = 1

        # lazy
        self.OBMol = OBMol
        '''
            Hbond definition sources: 
                1. "oddt" (https://github.com/oddt/oddt) 
                2. "lipinski" (https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Lipinski.py)
                3. "Vina"
        '''

        self.hbond_type = "lipinski"

        # call parent constructor
        super(Molecule, self).__init__(OBMol)

        self._protein = protein
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        self._coords = None
        self._charges = None

    @property
    def atoms(self):
        return AtomStack(self.OBMol)

    @property
    def bonds(self):
        return BondStack(self.OBMol)

    @property
    def coords(self):
        if self._coords is None:
            self._coords = np.array(
                [atom.coords for atom in self.atoms], dtype=np.float32)
            self._coords.setflags(write=False)
        return self._coords

    @property
    def atom_dict(self):
        # check cache and generate dicts
        if self._atom_dict is None:
            self._dicts()
        return self._atom_dict

    @property
    def res_dict(self):
        # check cache and generate dicts
        if self._res_dict is None:
            self._dicts()
        return self._res_dict

    @property
    def ring_dict(self):
        # check cache and generate dicts
        if self._ring_dict is None:
            self._dicts()
        return self._ring_dict

    @property
    def residues(self):
        return ResidueStack(self.OBMol)

    @property
    def protein(self):
        """
        A flag for identifing the protein molecules, for which `atom_dict`
        procedures may differ.
        """
        return self._protein

    @protein.setter
    def protein(self, protein):
        """atom_dict caches must be cleared due to property change"""
        self._clear_cache()
        self._protein = protein

    def _dicts(self):
        max_neighbors = 6  # max of 6 neighbors should be enough
        # Atoms
        atom_dtype = [('id', np.uint32),
                      # atom info
                      ('coords', np.float32, 3),
                      ('radius', np.float32),
                      ('charge', np.float32),
                      ('atomicnum', np.int8),
                      ('atomtype', 'U5' if PY3 else 'a5'),
                      ('hybridization', np.int8),
                      ('neighbors_id', np.int16, max_neighbors),
                      ('neighbors', np.float32, (max_neighbors, 3)),
                      # residue info
                      ('resid', np.int16),
                      ('resnum', np.int16),
                      ('resname', 'U3' if PY3 else 'a3'),
                      ('isbackbone', bool),
                      # atom properties
                      ('isacceptor', bool),
                      ('isdonor', bool),
                      ('isdonorh', bool),
                      ('ismetal', bool),
                      ('ishydrophobe', bool),
                      ('isaromatic', bool),
                      ('isminus', bool),
                      ('isplus', bool),
                      ('ishalogen', bool),
                      # secondary structure
                      ('isalpha', bool),
                      ('isbeta', bool)
                      ]

        atom_dict = np.empty(self.OBMol.NumAtoms(), dtype=atom_dtype)
        metals = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                  30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                  50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                  69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                  87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                  102, 103]
        for i, atom in enumerate(self.atoms):

            atomicnum = atom.atomicnum
            # skip non-polar hydrogens for performance
#            if atomicnum == 1 and atom.OBAtom.IsNonPolarHydrogen():
#                continue
            atomtype = typetable.Translate(atom.type)  # sybyl atom type
            partialcharge = atom.partialcharge
            coords = atom.coords

            if self.protein:
                residue = Residue(atom.OBAtom.GetResidue())
            else:
                residue = False

            # get neighbors, but only for those atoms which realy need them
            neighbors = np.zeros(max_neighbors, dtype=[('id', np.int16),
                                                       ('coords', np.float32, 3),
                                                       ('atomicnum', np.int8)])
            neighbors['coords'].fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                if n >= max_neighbors:
                    warnings.warn('Error while parsing molecule "%s" '
                                  'for `atom_dict`. Atom #%i (%s) has %i '
                                  'neighbors (max_neighbors=%i). Additional '
                                  'neighbors are ignored.' % (self.title,
                                                              atom.idx0,
                                                              atomtype,
                                                              len(atom.neighbors),
                                                              max_neighbors),
                                  UserWarning)
                    break
                if nbr_atom.atomicnum == 1:
                    # continue
                    neighbors[n] = (nbr_atom.idx0, nbr_atom.coords,
                                    nbr_atom.atomicnum)
            assert i == atom.idx0
            # for compatibability of openbabel 3 and 2

            def getVdwRad(atomicnum):
                try:
                    rad = ob.GetVdwRad(atomicnum)
                    return rad
                except:
                    rad = ob.OBElementTable().GetVdwRad(atomicnum)
                    return rad
            atom_dict[i] = (i,
                            coords,
                            # v2.4 ob.OBElementTable().GetVdwRad(atomicnum)
                            # ob.GetVdwRad(atomicnum),
                            getVdwRad(atomicnum),
                            partialcharge,
                            atomicnum,
                            atomtype,
                            atom.OBAtom.GetHyb(),
                            neighbors['id'],
                            neighbors['coords'],
                            # residue info
                            residue.idx0 if residue else 0,
                            residue.number if residue else 0,
                            residue.name if residue else '',
                            residue.OBResidue.GetAtomProperty(
                                atom.OBAtom, 2) if residue else False,  # is backbone
                            # atom properties
                            False,  # atom.OBAtom.IsHbondAcceptor(),
                            False,  # atom.OBAtom.IsHbondDonor(),
                            False,  # atom.OBAtom.IsHbondDonorH(),
                            atomicnum in metals,
                            atomicnum == 6 and np.in1d(neighbors['atomicnum'], [
                                                       6, 1, 0]).all(),  # hydrophobe
                            atom.OBAtom.IsAromatic(),
                            atom.formalcharge < 0,  # is charged (minus)
                            atom.formalcharge > 0,  # is charged (plus)
                            atomicnum in [9, 17, 35, 53],  # is halogen?
                            False,  # alpha
                            False  # beta
                            )

        not_carbon = np.argwhere(
            ~np.in1d(atom_dict['atomicnum'], [1, 6])).flatten()
        # Acceptors
        if self.hbond_type == "oddt":
            patt = Smarts('[$([O;H1;v2]),'
                          '$([O;H0;v2;!$(O=N-*),'
                          '$([O;-;!$(*-N=O)]),'
                          '$([o;+0])]),'
                          '$([n;+0;!X3;!$([n;H1](cc)cc),'
                          '$([$([N;H0]#[C&v4])]),'
                          '$([N&v3;H0;$(Nc)])]),'
                          '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
        elif self.hbond_type == "lipinski":
            patt = Smarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),'
                          '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),'
                          '$([nH0,o,s;+0])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isacceptor'][np.intersect1d(
                matches - 1, not_carbon)] = True

        # Donors
        if self.hbond_type == "oddt":
            patt = Smarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                          '$([$(n[n;H1]),'
                          '$(nc[n;H1])])]),'
                          # Guanidine can be tautormeic - e.g. Arginine
                          '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                          '$([O,S;H1;+0])]')
        elif self.hbond_type == "lipinski":
            # print("The definition of hbond was obtained from https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Lipinski.py")
            patt = Smarts(
                '[$([N;!H0;v3]),'
                '$([N;!H0;+1;v4]),'
                '$([O,S;H1;+0]),'
                '$([n;H1;+0])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isdonor'][np.intersect1d(
                matches - 1, not_carbon)] = True
            atom_dict['isdonorh'][[n.idx0
                                   for idx in np.argwhere(atom_dict['isdonor']).flatten()
                                   for n in self.atoms[int(idx)].neighbors
                                   if n.atomicnum == 1]] = True

        # Basic group
        patt = Smarts('[$([N;H2&+0][$([C,a]);!$([C,a](=O))]),'
                      '$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),'
                      '$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),'
                      '$([N,n;X2;+0])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isplus'][np.intersect1d(matches - 1, not_carbon)] = True

        # Acidic group
        patt = Smarts('[CX3](=O)[OX1H0-,OX2H1]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isminus'][np.intersect1d(
                matches - 1, not_carbon)] = True

        if self.protein:
            # Protein Residues (alpha helix and beta sheet)
            res_dtype = [('id', np.int16),
                         ('resnum', np.int16),
                         ('resname', 'U3' if PY3 else 'a3'),
                         ('N', np.float32, 3),
                         ('CA', np.float32, 3),
                         ('C', np.float32, 3),
                         ('O', np.float32, 3),
                         ('isalpha', bool),
                         ('isbeta', bool)
                         ]  # N, CA, C, O

            b = []
            for residue in self.residues:
                backbone = {}
                for atom in residue:
                    if residue.OBResidue.GetAtomProperty(atom.OBAtom, 1):
                        if atom.atomicnum == 7:
                            backbone['N'] = atom.coords
                        elif atom.atomicnum == 6:
                            if atom.type == 'C3':
                                backbone['CA'] = atom.coords
                            else:
                                backbone['C'] = atom.coords
                        elif atom.atomicnum == 8:
                            backbone['O'] = atom.coords
                if len(backbone.keys()) == 4:
                    b.append((residue.idx0,
                              residue.number,
                              residue.name,
                              backbone['N'],
                              backbone['CA'],
                              backbone['C'],
                              backbone['O'],
                              False,
                              False))
            res_dict = np.array(b, dtype=res_dtype)
            res_dict = detect_secondary_structure(res_dict)
            alpha_mask = np.in1d(atom_dict['resid'],
                                 res_dict[res_dict['isalpha']]['id'])
            atom_dict['isalpha'][alpha_mask] = True
            beta_mask = np.in1d(atom_dict['resid'],
                                res_dict[res_dict['isbeta']]['id'])
            atom_dict['isbeta'][beta_mask] = True

        # Aromatic Rings
        r = []
        for ring in self.sssr:

            if ring.IsAromatic():
                path = [x - 1 for x in ring._path]  # NOTE: mol.sssr is 1-based
                atoms = atom_dict[canonize_ring_path(path)]
                if len(atoms):
                    atom = atoms[0]
                    coords = atoms['coords']
                    centroid = coords.mean(axis=0)
                    # get vector perpendicular to ring
                    ring_vectors = coords - centroid
                    vector = np.cross(ring_vectors, np.roll(
                        ring_vectors, 1)).mean(axis=0)
                    # get atom id in ring
                    ring_atoms_id = np.zeros(10)-1
                    for idx in range(len(atoms)):
                        if idx < 10:
                            ring_atoms_id[idx] = atoms[idx]['id']
                    r.append((centroid,
                              vector,
                              atom['resid'],
                              atom['resnum'],
                              atom['resname'],
                              atom['isalpha'],
                              atom['isbeta'],
                              ring_atoms_id))
        ring_dict = np.array(r, dtype=[('centroid', np.float32, 3),
                                       ('vector', np.float32, 3),
                                       ('resid', np.int16),
                                       ('resnum', np.int16),
                                       ('resname', 'U3' if PY3 else 'a3'),
                                       ('isalpha', bool),
                                       ('isbeta', bool),
                                       ('atoms', np.int16, 10)])
        # if self.debug != 0:
        #     print(ring_dict)

        self._atom_dict = atom_dict
        self._atom_dict.setflags(write=False)
        self._ring_dict = ring_dict
        self._ring_dict.setflags(write=False)
        if self.protein:
            self._res_dict = res_dict
            self._res_dict.setflags(write=False)
        # if self.debug:
        #     for itm in range(len(atom_dict)):
        #         print("\nAtom Info of %s\n" % (itm))
        #         for jtm in range(24):
        #             print("%s:  %s" %
        #                   (atom_dtype[jtm][0], self._atom_dict[itm][jtm]))


class AtomStack(object):

    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumAtoms()):
            yield Atom(self.OBMol.GetAtom(i + 1))

    def __len__(self):
        return self.OBMol.NumAtoms()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumAtoms():
            return Atom(self.OBMol.GetAtom(int(i + 1)))
        else:
            raise AttributeError("There is no atom with Idx %i" % i)


class Atom(pybel.Atom):
    @property
    @deprecated('RDKit is 0-based and OpenBabel is 1-based. '
                'State which convention you desire and use `idx0` or `idx1`.')
    def idx(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        return self.idx1

    @property
    def idx1(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        return self.OBAtom.GetIdx()

    @property
    def idx0(self):
        """Note that this index is 0-based and OpenBabel's internal index in
        1-based. Changed to be compatible with RDKit"""
        return self.OBAtom.GetIdx() - 1

    @property
    def neighbors(self):
        return [Atom(a) for a in OBAtomAtomIter(self.OBAtom)]

    @property
    def residue(self):
        return Residue(self.OBAtom.GetResidue())

    @property
    def bonds(self):
        return [Bond(b) for b in OBAtomBondIter(self.OBAtom)]


class BondStack(object):
    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumBonds()):
            yield Bond(self.OBMol.GetBond(i))

    def __len__(self):
        return self.OBMol.NumBonds()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumBonds():
            return Bond(self.OBMol.GetBond(i))
        else:
            raise AttributeError("There is no bond with Idx %i" % i)


class Bond(object):
    def __init__(self, OBBond):
        self.OBBond = OBBond

    @property
    def order(self):
        return self.OBBond.GetBondOrder()

    @property
    def atoms(self):
        return (Atom(self.OBBond.GetBeginAtom()), Atom(self.OBBond.GetEndAtom()))

    @property
    def isrotor(self):
        return self.OBBond.IsRotor()


class Residue(object):
    """Represent a Pybel residue.

    Required parameter:
       OBResidue -- an Open Babel OBResidue

    Attributes:
       atoms, idx, name.

    (refer to the Open Babel library documentation for more info).

    The original Open Babel atom can be accessed using the attribute:
       OBResidue
    """

    def __init__(self, OBResidue):
        self.OBResidue = OBResidue

    @property
    def atoms(self):
        """List of Atoms in the Residue"""
        return [Atom(atom) for atom in ob.OBResidueAtomIter(self.OBResidue)]

    @property
    @deprecated('Use `idx0` instead.')
    def idx(self):
        """Internal index (0-based) of the Residue"""
        return self.OBResidue.GetIdx()

    @property
    def idx0(self):
        """Internal index (0-based) of the Residue"""
        return self.OBResidue.GetIdx()

    @property
    def number(self):
        """Residue number"""
        return self.OBResidue.GetNum()

    @property
    def chain(self):
        """Resdiue chain ID"""
        return self.OBResidue.GetChain()

    @property
    def name(self):
        """Residue name"""
        return self.OBResidue.GetName()

    def __iter__(self):
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print(atom)
        """
        return iter(self.atoms)


class ResidueStack(object):
    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumResidues()):
            yield Residue(self.OBMol.GetResidue(i))

    def __len__(self):
        return self.OBMol.NumResidues()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumResidues():
            return Residue(self.OBMol.GetResidue(i))
        else:
            raise AttributeError("There is no residue with Idx %i" % i)
