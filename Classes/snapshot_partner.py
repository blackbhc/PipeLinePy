import numpy as np
import scipy as sp
import h5py
from scipy.stats import binned_statistic as bin1d
from scipy.fftpack import fft

class single_snapshot_partner:
    """
    Util class to analysis single hdf5 snapshot file.

    Note: the code will use the internal units of the snapshot file.
    """
    def __init__(self, filename, dir='.', target_datasets = [], autoanalysis=True, info=True):
        """
        filename: str, the filename of the hdf5 snapshot file.

        dir: str, the directory of the snapshot file, default = "./".

        target_datasets: a list/tuple/numpy.array of strs, specify the interested datasets (PartType0, for example), 
        default=[].

        autoanalysis: automatically finish the analysis of the snapshot

        info: whether print the reminder Information.
        """

        # User uninsterested part
        self.__info = info
        if self.__info: print("Initializing the single snapshot partner object ...\n")
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
        self.__bar_major_axis = None # the azimuthal angle of major axis, in [rad]
        self.get_bar_major_axis = lambda: self.__bar_major_axis
        self.__bar_strength = None # the bar strength
        self.get_bar_strength = lambda: self.__bar_strength
        self.__bar_semi_length = None # the half bar length, in [kpc]
        self.get_bar_semi_length = lambda: self.__bar_semi_length
        self.__buckling_strength = None # the buckling strength
        self.get_buckling_strength = lambda: self.__buckling_strength
        self.__has_mass = False # whether has mass in the snapshot
        self.has_mass = lambda: self.__has_mass # API
        self.get_cylindrical_coordinates = lambda: self.__cylindrical_coordiantes

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
        
        for key in self.dataset_keys:
            try:
                self.snapshot[key]['Masses']
            except:
                pass
            else:
                self.__has_mass = True
                break
        
        if autoanalysis:
            # readin data
            self.readdata()
            self.recenter(sphere_size=100)
            self.calculate_cylindrical_coordinates()
            self.calculate_bar_strength()
            self.calculate_buckling_strength()
            self.annulus_fourier_analysis()
            self.calculate_bar_major_axis()

        # prompts
        if self.__info: print("Initialization done!\n")


    def readdata(self, target_datasets=None):
        """
        Read in the data of interested datasets (specified by target_datatets) from snapshot files.
        Note: you can overwrite the target_datasets in this function, if you give a none empty value to it at here.
        """
        # prompts
        if self.__info: print("Reading in the target datasets ...")

        if (target_datasets):
            # defensive part: check whether there is(are) the target dataset(s)
            for target in self.target_datasets:
                if not(target in self.dataset_keys):
                    raise KeyError( "dataset <{}> not found in the file <{}>".format(target, dir+'/'+filename) ) 
            self.target_datasets = target_datasets
            
        self.data = []
        for target in self.target_datasets:
            self.data.append(self.snapshot[target])

        # read in the coordinates and masses target particles
        self.__coordinates = []
        if self.has_mass(): self.__masses = []
        for subset in self.data:
            self.__coordinates.append(subset["Coordinates"][...])
            if self.has_mass(): self.__masses.append(subset["Masses"][...])
        self.__coordinates = np.row_stack(self.__coordinates)
        if self.has_mass(): self.__masses = np.squeeze(np.column_stack(self.__masses))

        if self.has_potential():
            self.__potentials = []
            for subset in self.data:
                self.__potentials.append(subset["Potential"][...])
            self.__potentials = np.squeeze(np.column_stack(self.__potentials))
        # read in potentials if the snapshot has potential information

        # prompts
        if self.__info: print("Read in datasets done!\n")


    #def recenter(self, sphere_size=None, box_size=None, MAXLOOP=1000):
    def recenter(self, sphere_size=100, box_size=None, MAXLOOP=1000): #debug
        """
        Method to recenter the system (set of all target particles).
        
        sphere_size: specify the region of calculation of recenter, in this case it's a sphere.
        
        box_size: if sphere_size=None, then use this size to specify a cube aroud the __system_center,
        of which the width=box_size.
        Note: if the system has Potential, the CoM is of the most bounded particles (1%), otherwise it's the one
        of all particles in the specified region.

        MAXLOOP: the max times of the loop for recentering
        """
        # prompts
        if self.__info: print("Recentering the system ...")
        self.recentered = False; self.get_recenter_status = lambda: self.recentered # whether have recentered

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
                index = np.where( np.linalg.norm(self.__coordinates - self.__system_center, axis=1, ord=2) <= sphere_size)[0]
                # index: inside a sphere
            elif box_size:
                index = np.where( np.abs( (self.__coordinates - self.__system_center)[:, 0] ) <= box_size/2)[0]
                index = index[np.where( np.abs( (self.__coordinates - self.__system_center)[index, 1] ) <= box_size/2)[0]]
                index = index[np.where( np.abs( (self.__coordinates - self.__system_center)[index, 2] ) <= box_size/2)[0]]
            else:
                index = [i for i in range(len(self.__coordinates))] # the full index

            if self.has_potential():
                potentials_sorted = self.__potentials*1.0
                potentials_sorted.sort()
                bounded_boundary = potentials_sorted[ int( len(potentials_sorted) * 0.01 ) - 1 ]
                # the boundary of the most bounded particles
                index = np.array(index)[ np.where( self.__potentials[index]<=bounded_boundary )[0] ]

            if self.has_mass():
                cur_coord = self.__coordinates[index, :]
                cur_mass = self.__masses[index]
                new = np.sum( cur_coord * np.column_stack( (cur_mass, cur_mass, cur_mass) ), axis = 0 ) / np.sum(cur_mass)
            else:
                cur_coord = self.__coordinates[index, :]
                new = np.sum( cur_coord , axis = 0 ) / len(cur_coord)

            self.__system_center = new
            variation = np.linalg.norm(new - old, ord=2)

            loop += 1
        # whether convergent
        if loop == MAXLOOP:
            print("""Warning: the recenter process has not converged yet! You may need to modify the value of
                  sphere_size or box_size or MAXLOOP.""")
            self.__recenter_convergent = False
        else:
            self.__recenter_convergent = True
        self.recenter_converges_yes = lambda: self.__recenter_convergent
        # the API to return the convergence status of the recentering
        self.recentered = True

        # prompts
        if self.__info: print("Recentering done!\n")


    def calculate_cylindrical_coordinates(self):
        """
        Calculate the cylindrical coordinates of the system particles, put them into self.__cylindrical_coordiantes.

        Note: the cylindrical coordiantes are all w.r.t to the system center after recentering.
        """
        # prompts
        if self.__info: print("Calculating the cylindrical coordinates of the system ...")
        if not(self.get_recenter_status): self.recenter() # recenter the system if haven't done it

        Rs = np.linalg.norm((self.__coordinates - self.__system_center)[:, :2], axis=1, ord=2) # R of (R, Phi, z)
        
        if 0 in Rs:
            # check whether there is a particle at the origin (system center), which means R<1e-3
            index = np.where( Rs>1e-3 )[0]
            Phis = np.arcsin( (self.__coordinates[index, 1]-self.__system_center[1])/Rs[index] )
            Phis = list(Phis)
            for id in np.where(Rs<=1e-3)[0]:
                Phis.insert(id, np.random.rand()*np.pi*2)
                # deal with R=0 particle(s) with random azimuthal angles
            Phis = np.arctan2( self.__coordinates[index, 1] - self.__system_center[1], self.__coordinates[index, 0] - self.__system_center[0])
        else:
            Phis = np.arctan2( self.__coordinates[:, 1] - self.__system_center[1], self.__coordinates[:, 0] - self.__system_center[0])

        Phis[ np.where(Phis<0)[0] ] += np.pi*2 # normalized the range to [0, 2pi]

        self.__cylindrical_coordiantes = np.column_stack((Rs, Phis, self.__coordinates[:, 2]-self.__system_center[2]))

        # prompts
        if self.__info: print("Calculate cylindrical coordiantes done!\n")


    def calculate_bar_length(self, disk_size=None):
        """
        Calculate the bar length.

        disk_size: only particles cylindrical radius < disk_size are quantified.
        """


    def calculate_bar_strength(self, region_size=4):
        """
        Calculate the bar strength parameter.

        region_size: only particles with cylindrical radius < region_size are quantified.
        """
        index = np.where( self.__cylindrical_coordiantes[:, 0] < region_size )[0]
        
        if self.has_mass():
            numerator = (self.__masses[index] * np.exp(2j * self.__cylindrical_coordiantes[index, 1])).sum()
            denominator = self.__masses[index].sum()
        else:
            numerator = (np.exp(2j * self.__cylindrical_coordiantes[index, 1])).sum()
            denominator = len(index)

        self.__bar_strength = abs(numerator / denominator)
        return self.get_bar_strength(), np.angle(numerator)

    
    def calculate_buckling_strength(self, region_size=10):
        """
        Calculate the buckling strength parameter.

        region_size: only particles with cylindrical radius < region_size are quantified.
        """
        index = np.where( self.__cylindrical_coordiantes[:, 0] < region_size )[0]
        
        if self.has_mass():
            numerator = (self.__cylindrical_coordiantes[index, 2] * self.__masses[index] *
                         np.exp(2j * self.__cylindrical_coordiantes[index, 1])).sum()
            denominator = self.__masses[index].sum()
        else:
            numerator = (self.__cylindrical_coordiantes[index, 2] *
                         np.exp(2j * self.__cylindrical_coordiantes[index, 1])).sum()
            denominator = len(index)

        self.__buckling_strength = abs(numerator / denominator)
        return self.get_buckling_strength()


    #  def annulus_fourier_analysis(self, minradius=0.0, maxradius=15, bins=21, inLgbins=False, azimuthal_bins=36):
    #      """
    #      Calculate the fourier coefficient A2 and it's phase angle for annuluses in the face-on image of the system.
    #
    #      minradius: the inner most radius for the calculation.
    #
    #      maxradius: the outer most radius for the calculation.
    #
    #      bins: binnum of the annuluses during calculation.
    #
    #      inLgbins: if True, the bins will be linear bewteen lg(minradius) and lg(maxradius)
    #
    #      azimuthal_bins: binnum of azimuthal bins for different azimuthal_bins
    #      """
    #      # prompts
    #      if self.__info :
    #          print("Fourier analysis of different circular annulus ...", sep=' ')
    #          if inLgbins:
    #              print("(Annuluses are in logarithmically linear bins)")
    #          else:
    #              print("(Annuluses are in linear bins.)")
    #
    #      # defensive part
    #      if minradius==0 and inLgbins:
    #          minradius = 1e-1
    #          print(f"Lg(0) is not possible, the program will use {minradius} as inner most raduis instead 0 ...")
    #
    #      # if inLgbins
    #      if inLgbins:
    #          minradius = np.log10(minradius)
    #          maxradius = np.log10(maxradius)
    #
    #      bin_edges = np.linspace(minradius, maxradius, bins)
    #      if inLgbins: bin_edges = 10**(bin_edges)
    #      indexes = []
    #      for i in range(len(bin_edges)-1):
    #          index = np.where(self.__cylindrical_coordiantes[:, 0] >= bin_edges[i])[0]
    #          index = np.array(index)[np.where(self.__cylindrical_coordiantes[index, 0] < bin_edges[i+1])[0]]
    #          indexes.append(index)
    #
    #      # calculation part
    #      A2 = []; Angles = [] # the A2 and major axis of different annuluses
    #      for id in indexes:
    #          # defensive: case that there is no particle in some inner/outer most bins
    #          if len(id)==0:
    #              A2.append(None)
    #              angles.append(None)
    #              continue
    #
    #          phis = self.__cylindrical_coordiantes[id, 1]
    #          column_densitys, _, _ = bin1d(x = phis, values = phis, bins =\
    #                  azimuthal_bins, statistic = "count")
    #          column_densitys = self.__lg_norm1d(column_densitys)
    #          column_densitys -= column_densitys.mean()
    #          Freqs, results, angles = self.__simple_fourier_analysis(data = column_densitys)
    #          A2.append(results[2])
    #          Angles.append(angles[2])
    #
    #      self.__A2 = np.array(A2)
    #      self.__m2_major_axis = np.array(Angles)
    #      self.get_A2 = lambda: self.__A2
    #      self.get_m2_major_axis = lambda: self.__m2_major_axis
    #
    #      # prompts
    #      if self.__info : print("Fourier analysis of different circular annulus is done!")
    #
    #
    #  def calculate_bar_major_axis(self, region_size=4, binnum=36):
    #      """
    #      Calculate the bar major axis in a cylindrical region as the m2 sysmetry mode's major axis.
    #
    #      region_size: radius of such cylindrical reigon.
    #
    #      binnum: number of bins for Fourier analysis.
    #      """
    #      index = np.where( self.__cylindrical_coordiantes[:, 0] <= region_size)[0]
    #      index = index[np.where( self.__cylindrical_coordiantes[index, 2] <= 0.25)[0]]
    #      column_densitys, _,_ = bin1d(x = self.__cylindrical_coordiantes[index, 1], values = \
    #              self.__cylindrical_coordiantes[index, 1], statistic = "count", bins=binnum,)
    #
    #      column_densitys = self.__lg_norm1d(column_densitys)
    #      _, _, angle = self.__simple_fourier_analysis(data = column_densitys)
    #
    #      self.__bar_major_axis = angle[2]
    #
    #  # some utils functions
    #  def __lg_norm1d(self, array):
    #      data = array*1.0
    #      index = np.where(data<1)[0]
    #      data[index] = 1
    #      data = np.log10(data)
    #      return data
    #
    #  def __simple_fourier_analysis(self, data):
    #      N = len(data)
    #      results = np.abs( fft(x = data)) / N * 2 # normalization
    #      angles = np.angle( fft(x = data))
    #      results[0] /= 2 # further normalization of A0
    #      Freqs = np.arange(int(N / 2))
    #      results = results[range(int(N / 2))] # only positive side
    #      angles = angles[range(int(N / 2))] # only positive side
    #      return Freqs, results, angles
    def calculate_major_axis(self, region_size)



class snapshots_partner:
    """
    Analysis tool of multiple snapshots, based on the class single_snapshot_partner.
    """
    def __init__(self, filenames, dir=".", target_datasets=[], info=False):
        """
        filenames: list/tuple/numpy.array of strings, which contains the filenames of the snapshots.

        dir: string, directory of the snapshot files, default=current directory.

        target_datasets: a list/tuple/numpy.array of strs, specify the interested datasets (PartType0, for example), 
        default=[].

        info: whether print the reminder message, as there are many snapshots, default=False
        """
        self.__snapshots = [] # list of the snapshots
        
        for i in filenames:
            self.__snapshots.append(single_snapshot_partner(filename = i, dir=dir, target_datasets=target_datasets,\
                    autoanalysis = True, info=info));

        self.__bar_strengths = [snapshot.get_bar_strength() for snapshot in self.__snapshots] # bar strength parameters
        self.get_bar_strengths = lambda: tuple(self.__bar_strengths) # secure API to return bar strength parameters
        self.__buckling_strengths = [snapshot.get_buckling_strength() for snapshot in self.__snapshots]
        self.get_buckling_strengths = lambda: self.__buckling_strengths




