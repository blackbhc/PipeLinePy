import numpy as np
import scipy as sp
import h5py

class single_snapshot_partner:
    """
    Util class to analysis single hdf5 snapshot file.
    """
    def __init__(self, filename, dir='.', target_datasets = [], autoanalysis=True):
        """
        filename: str, the filename of the hdf5 snapshot file.

        dir: str, the directory of the snapshot file, default = "./".

        target_datasets: a list/tuple of strs, specify the interested datasets (PartType0, for example), default=[].

        autoanalysis: automatically finish the analysis of the snapshot
        """
        # defensive part: check whether there is the target file
        try:
            self.snapshot = h5py.File(dir+'/'+filename, 'r')
        except:
            print(dir+"/"+filename, ": Not found!")
        else:
            # the real data
            self.dataset_keys = self.snapshot.keys()
            # the keys of the available datasets
            self.target_datasets = target_datasets
            # the interested dataset(s) of the snapshot

        # defensive part: check whether there is(are) the target dataset(s)
        for target in self.target_datasets:
            if not(target in self.dataset_keys):
                raise KeyError( "dataset <{}> not found in the file <{}>".format(target, dir+'/'+filename) ) 

        # the model statistical quantifications
        self.__system_center = np.array([0, 0, 0]) # the center of the system
        self.get_system_center = lambda: self.__system_center # API to get the center of the system
        self.__major_axis = 0 # the azimuthal angle of major axis, in [rad]
        self.get_major_axis = lambda: self.__major_axis
        self.__bar_strength = 0 # the bar strength
        self.get_bar_strength = lambda: self.__bar_strength
        self.__bar_semi_length = 0 # the half bar length, in [kpc]
        self.get_bar_semi_length = lambda: self.__bar_semi_length
        self.__buckling_strength = 0 # the buckling strength
        self.get_buckling_strenth = lambda: self.__buckling_strength

        # data check: whether potential and OtF data (Unfinished!!!!!)
        self.__has_potential = False # check whther there are potential datasets
        self.has_potential = lambda: self.__has_potential # API to get info of potential
        for key in self.dataset_keys:
            try:
                self.snapshot[key]['Potential']
            except:
                pass
            else:
                self.__has_potential = True
                break
        
        if autoanalysis:
            # readin data
            self.readdata()
            self.recenter()


    def readdata(self, target_datasets=None):
        """
        Read in the data of interested datasets (specified by target_datatets) from snapshot files.
        Note: you can overwrite the target_datasets in this function, if you give a none empty value to it at here.
        """
        if (target_datasets):
            # defensive part: check whether there is(are) the target dataset(s)
            for target in self.target_datasets:
                if not(target in self.dataset_keys):
                    raise KeyError( "dataset <{}> not found in the file <{}>".format(target, dir+'/'+filename) ) 
            self.target_datasets = target_datasets
            

        self.data = []
        for target in self.target_datasets:
            self.data.append(self.snapshot[target])


    def recenter(self, sphere_size=None, box_size=None, MAXLOOP=1000):
        """
        Method to recenter the system (set of all target particles).
        
        sphere_size: specify the region of calculation of recenter, in this case it's a sphere.
        
        box_size: if sphere_size=None, then use this size to specify a cube aroud the __system_center,
        of which the width=box_size.
        Note: if the system has Potential, the CoM is of the most bounded particles (1%), otherwise it's the one
        of all particles in the specified region.

        MAXLOOP: the max times of the loop for recentering
        """
        coordinates = []; masses = []
        for subset in self.data:
            coordinates.append(subset["Coordinates"][...])
            masses.append(subset["Masses"][...])
        coordinates = np.row_stack(coordinates)
        masses = np.squeeze(np.column_stack(masses))
        # use the coordinates and masses of targer particles

        if self.has_potential():
            potentials = []
            for subset in self.data:
                potentials.append(subset["Potential"][...])
            potentials = np.squeeze(np.column_stack(potentials))
        # read in potentials if the snapshot has potential information

        variation = 1000
        # the variation of the system center in [kpc] (default base unit of the distance in the snapshot)
        if sphere_size:
            criterion = sphere_size * 0.001
        elif box_size:
            criterion = box_size * 0.001
        else:
            criterion = 0.1
        # the criterion to stop the loop
        loop = 1
        while(variation > criterion and loop < MAXLOOP):
            old = self.__system_center
            # calculate the index of the region, which will be used to calculate the symtem center
            if sphere_size:
                index = np.where( np.linalg.norm(coordinates - self.__system_center, axis=1, ord=2) <= sphere_size)[0]
                # index: inside a sphere
            elif box_size:
                index = np.where( np.abs( (coordinates - self.__system_center)[:, 0] ) <= box_size/2)[0]
                index = index[np.where( np.abs( (coordinates - self.__system_center)[index, 1] ) <= box_size/2)[0]]
                index = index[np.where( np.abs( (coordinates - self.__system_center)[index, 2] ) <= box_size/2)[0]]
            else:
                index = [i for i in range(len(masses))] # the full index

            if self.has_potential():
                potentials_sorted = potentials*1.0
                potentials_sorted.sort()
                bounded_boundary = potentials_sorted[ int( len(potentials_sorted) * 0.01 ) - 1 ]
                # the boundary of the most bounded particles
                index = np.array(index)[ np.where( potentials[index]<=bounded_boundary )[0] ]

            cur_coord = coordinates[index, :]
            cur_mass = masses[index]
            new = np.sum( cur_coord * np.column_stack( (cur_mass, cur_mass, cur_mass) ), axis = 0 ) / np.sum(cur_mass)
            self.__system_center = new
            variation = np.linalg.norm(new - old, ord=2)

            loop += 1
        # whether convergent
        if loop == MAXLOOP:
            print("""Warning: the recenter process has not converged yet! You may need to modify the value of
                  sphere_size or box_size or MAXLOOP.""")

    def calculate_bar_length(self, disk_size=None):
        """
        Calculate the bar length.

        disk_size: only particles cylindrical radius < disk_size are quantified.
        """


    def calculate_bar_strength(self):
        """
        Calculate the bar strength parameter.
        """
