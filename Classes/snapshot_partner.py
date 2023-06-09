# TODO: add in an awosome process bar.
# TODO: add in the plotting parts.
# TODO: reconstruct the code to make it more readable: based on the scheme, and sperate the atributes and methods.
import os
import numpy as np
import scipy as sp
import h5py
from scipy.stats import binned_statistic as bin1d
from scipy.stats import binned_statistic_2d as bin2d
import imageio


class single_snapshot_partner:
    """
    Util class to analysis single hdf5 snapshot file.

    Note: the code will use the internal units of the snapshot file.
    """

    def __init__(
        self, filename, dir=".", target_datasets=[], autoanalysis=True, info=True
    ):
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
        if self.__info:
            print("Initializing the single snapshot partner object ...\n")
        # defensive part: check whether there is the target file
        try:
            self.snapshot = h5py.File(dir + "/" + filename, "r")
        except:
            print(dir + "/" + filename, ": Not found!")
        else:
            # the real data
            self.dataset_keys = self.snapshot.keys()
            # the keys of the available datasets
            self.target_datasets = target_datasets
            # the interested dataset(s) of the snapshot

        # defensive part: check whether there is(are) the target dataset(s)
        for target in self.target_datasets:
            if not (target in self.dataset_keys):
                raise KeyError(
                    "dataset <{}> not found in the file <{}>".format(
                        target, dir + "/" + filename
                    )
                )

        # the model statistical quantifications
        self.__system_center = np.array([0, 0, 0])  # the center of the system
        self.get_system_center = (
            lambda: self.__system_center
        )  # API to get the center of the system
        self.__bar_major_axis = None  # the azimuthal angle of major axis, in [rad]
        self.get_bar_major_axis = (
            lambda region_size=3, method=1: self.calculate_bar_major_axis(
                region_size, method
            )
        )
        self.__bar_strength = None  # the bar strength
        self.get_bar_strength = lambda: self.__bar_strength
        self.__bar_semi_length = None  # the half bar length, in [kpc]
        self.get_bar_semi_length = lambda: self.__bar_semi_length
        self.__buckling_strength = None  # the buckling strength
        self.get_buckling_strength = lambda: self.__buckling_strength
        self.__has_mass = False  # whether has mass in the snapshot
        self.has_mass = lambda: self.__has_mass  # API
        self.get_cylindrical_coordinates = lambda: self.__cylindrical_coordiantes
        self.get_coordinates = lambda: self.__coordinates  # API

        # data check: whether potential and OtF data (Unfinished!!!!!)
        self.__has_potential = False  # check whther there are potential datasets
        self.has_potential = (
            lambda: self.__has_potential
        )  # API to get info of potential
        for key in self.dataset_keys:
            try:
                self.snapshot[key]["Potential"]
            except:
                pass
            else:
                self.__has_potential = True
                break

        for key in self.dataset_keys:
            try:
                self.snapshot[key]["Masses"]
            except:
                pass
            else:
                self.__has_mass = True
                break

        if autoanalysis:
            # readin data
            self.readdata()
            self.recenter(sphere_size=1000)
            self.calculate_cylindrical_coordinates()
            self.calculate_bar_strength()
            self.calculate_buckling_strength()
            self.calculate_bar_major_axis()

        # prompts
        if self.__info:
            print("Initialization done!\n")

    def readdata(self, target_datasets=None):
        """
        Read in the data of interested datasets (specified by target_datatets) from snapshot files.
        Note: you can overwrite the target_datasets in this function, if you give a none empty value to it at here.
        """
        # prompts
        if self.__info:
            print("Reading in the target datasets ...")

        if target_datasets:
            # defensive part: check whether there is(are) the target dataset(s)
            for target in self.target_datasets:
                if not (target in self.dataset_keys):
                    raise KeyError(
                        "dataset <{}> not found in the file <{}>".format(
                            target, dir + "/" + filename
                        )
                    )
            self.target_datasets = target_datasets

        self.data = []
        for target in self.target_datasets:
            self.data.append(self.snapshot[target])

        # read in the coordinates and masses target particles
        self.__coordinates = []
        if self.has_mass():
            self.__masses = []
        for subset in self.data:
            self.__coordinates.append(subset["Coordinates"][...])
            if self.has_mass():
                self.__masses.append(subset["Masses"][...])
        self.__coordinates = np.row_stack(self.__coordinates)
        if self.has_mass():
            self.__masses = np.squeeze(np.column_stack(self.__masses))

        if self.has_potential():
            self.__potentials = []
            for subset in self.data:
                self.__potentials.append(subset["Potential"][...])
            self.__potentials = np.squeeze(np.column_stack(self.__potentials))
        # read in potentials if the snapshot has potential information

        # prompts
        if self.__info:
            print("Read in datasets done!\n")

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
        # prompts
        if self.__info:
            print("Recentering the system ...")
        self.recentered = False
        self.get_recenter_status = lambda: self.recentered  # whether have recentered

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
        while variation > criterion and loop < MAXLOOP:
            old = self.__system_center
            # calculate the index of the region, which will be used to calculate the symtem center
            if sphere_size:
                index = np.where(
                    np.linalg.norm(
                        self.__coordinates - self.__system_center, axis=1, ord=2
                    )
                    <= sphere_size
                )[0]
                # index: inside a sphere
            elif box_size:
                index = np.where(
                    np.abs((self.__coordinates - self.__system_center)[:, 0])
                    <= box_size / 2
                )[0]
                index = index[
                    np.where(
                        np.abs((self.__coordinates - self.__system_center)[index, 1])
                        <= box_size / 2
                    )[0]
                ]
                index = index[
                    np.where(
                        np.abs((self.__coordinates - self.__system_center)[index, 2])
                        <= box_size / 2
                    )[0]
                ]
            else:
                index = [i for i in range(len(self.__coordinates))]  # the full index

            if self.has_potential():
                potentials_sorted = self.__potentials * 1.0
                potentials_sorted.sort()
                bounded_boundary = potentials_sorted[
                    int(len(potentials_sorted) * 0.01) - 1
                ]
                # the boundary of the most bounded particles
                index = np.array(index)[
                    np.where(self.__potentials[index] <= bounded_boundary)[0]
                ]

            if self.has_mass():
                cur_coord = self.__coordinates[index, :]
                cur_mass = self.__masses[index]
                new = np.sum(
                    cur_coord * np.column_stack((cur_mass, cur_mass, cur_mass)), axis=0
                ) / np.sum(cur_mass)
            else:
                cur_coord = self.__coordinates[index, :]
                new = np.sum(cur_coord, axis=0) / len(cur_coord)

            self.__system_center = new
            variation = np.linalg.norm(new - old, ord=2)

            loop += 1
        # whether convergent
        if loop == MAXLOOP:
            print(
                """Warning: the recenter process has not converged yet! You may need to modify the value of
                  sphere_size or box_size or MAXLOOP."""
            )
            self.__recenter_convergent = False
        else:
            self.__recenter_convergent = True
        self.recenter_converges_yes = lambda: self.__recenter_convergent
        # the API to return the convergence status of the recentering
        self.recentered = True

        # prompts
        if self.__info:
            print("Recentering done!\n")

    def calculate_cylindrical_coordinates(self):
        """
        Calculate the cylindrical coordinates of the system particles, put them into self.__cylindrical_coordiantes.

        Note: the cylindrical coordiantes are all w.r.t to the system center after recentering.
        """
        # prompts
        if self.__info:
            print("Calculating the cylindrical coordinates of the system ...")
        if not (self.get_recenter_status):
            self.recenter()  # recenter the system if haven't done it

        Rs = np.linalg.norm(
            (self.__coordinates - self.__system_center)[:, :2], axis=1, ord=2
        )  # R of (R, Phi, z)

        if 0 in Rs:
            # check whether there is a particle at the origin (system center), which means R<1e-3
            index = np.where(Rs > 1e-3)[0]
            Phis = np.arcsin(
                (self.__coordinates[index, 1] - self.__system_center[1]) / Rs[index]
            )
            Phis = list(Phis)
            for id in np.where(Rs <= 1e-3)[0]:
                Phis.insert(id, np.random.rand() * np.pi * 2)
                # deal with R=0 particle(s) with random azimuthal angles
            Phis = np.arctan2(
                self.__coordinates[index, 1] - self.__system_center[1],
                self.__coordinates[index, 0] - self.__system_center[0],
            )
        else:
            Phis = np.arctan2(
                self.__coordinates[:, 1] - self.__system_center[1],
                self.__coordinates[:, 0] - self.__system_center[0],
            )

        Phis[np.where(Phis < 0)[0]] += np.pi * 2  # normalized the range to [0, 2pi]

        self.__cylindrical_coordiantes = np.column_stack(
            (Rs, Phis, self.__coordinates[:, 2] - self.__system_center[2])
        )

        # prompts
        if self.__info:
            print("Calculate cylindrical coordiantes done!\n")

    def calculate_bar_length(self, disk_size=None):
        """
        Calculate the bar length.

        disk_size: only particles cylindrical radius < disk_size are quantified.
        """

    def calculate_bar_strength(self, region_size=4):
        """
        Calculate the bar strength parameter.

        region_size: only particles with cylindrical radius < region_size are quantified.

        --------
        Returns:

        bar_strength: dimensionless bar strength parameter.
        """
        index = np.where(self.__cylindrical_coordiantes[:, 0] < region_size)[0]

        if self.has_mass():
            numerator = (
                self.__masses[index]
                * np.exp(2j * self.__cylindrical_coordiantes[index, 1])
            ).sum()
            denominator = self.__masses[index].sum()
        else:
            numerator = (np.exp(2j * self.__cylindrical_coordiantes[index, 1])).sum()
            denominator = len(index)

        self.__bar_strength = abs(numerator / denominator)
        return self.get_bar_strength()

    def calculate_buckling_strength(self, region_size=4):
        """
        Calculate the buckling strength parameter.

        region_size: only particles with cylindrical radius < region_size are quantified.

        --------
        Returns:

        buckling_strength: the buckling strength dimensionless parameter.
        """
        index = np.where(self.__cylindrical_coordiantes[:, 0] < region_size)[0]

        if self.has_mass():
            numerator = (
                self.__cylindrical_coordiantes[index, 2]
                * self.__masses[index]
                * np.exp(2j * self.__cylindrical_coordiantes[index, 1])
            ).sum()
            denominator = self.__masses[index].sum()
        else:
            numerator = (
                self.__cylindrical_coordiantes[index, 2]
                * np.exp(2j * self.__cylindrical_coordiantes[index, 1])
            ).sum()
            denominator = len(index)

        self.__buckling_strength = abs(numerator / denominator)
        return self.get_buckling_strength()

    def calculate_bar_major_axis(self, region_size=3, method=1):
        """
        Calculate the bar major axis of the system, which is the azimuthal direction of maximal particles.

        region_size: double, specify the region used in calculation, only particles that R<region_size are
        included in calculation, and 0.5~1 semi major axis of the bar is a recommended value.

        method: method used to calculate the bar major axis, must be one of (0, 1).
        0: major axis anlge:=the argument angle of the m=2 symmetry component of the mass distribution, namely A2.
        1: major axis angle:=the eigen vector of the inertia tensor of the system which has the maximal eigenvalue.

        --------
        Returns:

        phi: the azimuthal angle of bar major axis, in unit [rad]
        """
        # prompts
        if self.__info:
            print("Calculating the bar major axis in the X-Y plane ...")

        # defensive programming
        if method not in [0, 1]:
            raise ValueError("method must be one of (0, 1)!")

        # check whether the cylindrical coordinates are calculated
        try:
            self.__cylindrical_coordiantes
        except:
            self.calculate_cylindrical_coordinates()

        index = np.where(self.__cylindrical_coordiantes[:, 0] < region_size)[0]
        # index of particles in the target region

        if method == 0:  # major axis angle := arg(A2)
            # the similar code as in calculate_bar_strength, but here we only need numerator (A2)
            if self.has_mass():
                numerator = (
                    self.__masses[index]
                    * np.exp(2j * self.__cylindrical_coordiantes[index, 1])
                ).sum()
            else:
                numerator = (
                    np.exp(2j * self.__cylindrical_coordiantes[index, 1])
                ).sum()
            self.__bar_major_axis = (
                np.angle(numerator) / 2
            )  # the azimuthal angle of bar major axis
        else:
            if not (self.get_recenter_status):
                self.recenter(sphere_size=1000)
            pos = self.get_coordinates() - self.get_system_center()

            # calculate the inertia tensor
            inertia_tensor = np.zeros((3, 3))
            inertia_tensor[0, 0] = np.sum(pos[index, 1] ** 2 + pos[index, 2] ** 2)
            inertia_tensor[1, 1] = np.sum(pos[index, 0] ** 2 + pos[index, 2] ** 2)
            inertia_tensor[2, 2] = np.sum(pos[index, 0] ** 2 + pos[index, 1] ** 2)
            inertia_tensor[0, 1] = -np.sum(pos[index, 0] * pos[index, 1])
            inertia_tensor[0, 2] = -np.sum(pos[index, 0] * pos[index, 2])
            inertia_tensor[1, 2] = -np.sum(pos[index, 1] * pos[index, 2])
            inertia_tensor[1, 0] = inertia_tensor[0, 1]
            inertia_tensor[2, 0] = inertia_tensor[0, 2]
            inertia_tensor[2, 1] = inertia_tensor[1, 2]
            eigval, eigvec = np.linalg.eig(inertia_tensor)
            print(eigval)
            id = np.argmin(eigval)  # calculate the maximal eigenvalue
            self.__bar_major_axis = np.arctan2(
                eigvec[1, id], eigvec[0, id]
            )  # the azimuthal angle of bar major axis

        if self.__info:
            print("Calculation of bar major axis finished!")
        return self.__bar_major_axis

    def __exclude_one_sigma(self, data):
        """
        Exclude the data points of the input 1-D outside the one sigma region.
        """
        data = np.array(data)
        if len(data.shape) != 1:
            raise ValueError("Only allow 1-D array in this function(exclude_one_sigma)")
        mean = np.mean(data)
        std = np.std(data)
        index = np.where(data >= mean - std)[0]
        index = index[np.where(data[index] <= mean + std)[0]]
        return data[index]


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
        self.__snapshots = []  # list of the snapshots
        self.__info = info
        # prompts
        if self.__info:
            print("Initializating the snapshots partner ...")

        for i in filenames:
            self.__snapshots.append(
                single_snapshot_partner(
                    filename=i,
                    dir=dir,
                    target_datasets=target_datasets,
                    autoanalysis=True,
                    info=info,
                )
            )

        self.__bar_strengths = tuple(
            snapshot.get_bar_strength() for snapshot in self.__snapshots
        )  # bar strength parameters
        self.get_bar_strengths = (
            lambda: self.__bar_strengths
        )  # secure API to return bar strength parameters
        self.__buckling_strengths = tuple(
            snapshot.get_buckling_strength() for snapshot in self.__snapshots
        )  # buckling strengths
        self.get_buckling_strengths = lambda: self.__buckling_strengths
        self.__bar_major_axes = tuple(
            snapshot.get_bar_major_axis() for snapshot in self.__snapshots
        )  # bar major axis
        self.get_bar_major_axes = lambda: self.__bar_major_axes
        self.get_snapshot_count = lambda: len(self.__snapshots)
        self.__bar_pattern_speeds = None  # pattern speeds of the bar
        self.get_bar_pattern_speeds = lambda: self.__bar_pattern_speeds

        if self.__info:
            print("Initialization has been finished!")

    def calculate_bar_pattern_speeds(self, start=0, end=None):
        """
        Calculate the bar pattern speed with the azimuthal angles of the major axis.

        start:double, time stamp of the first snapshot, 0 by default.

        end: double, time stamp of the last snapshot, end=number of snapshots - 1 if not given.

        --------
        Returns:

        bar_pattern_speeds: n-1 double numpy.array, n is the count of snapshots, in unit [rad/time]
        """
        if self.__info:
            print("Calculating the pattern speed of the snapshots ...")
        if not (end):
            end = self.get_snapshot_count() - 1

        times = (
            np.arange(self.get_snapshot_count())
            / (self.get_snapshot_count() - 1)
            * (end - start)
            + start
        )
        age_bin = (end - start) / (self.get_snapshot_count() - 1)

        major_axis = np.array(self.get_bar_major_axes())
        delta_phis = major_axis[1:] - major_axis[:-1]
        bar_pattern_speeds = delta_phis / age_bin

        index_positive = np.where(bar_pattern_speeds > 0)[0]
        index_negative = np.where(bar_pattern_speeds <= 0)[0]
        N_positive = len(index_positive)
        N_negative = len(index_negative)
        if N_positive >= N_negative:
            # anticlockwise rotation
            bar_pattern_speeds[index_negative] = (
                delta_phis[index_negative] + np.pi
            ) / age_bin
        else:
            # clockwise ration
            bar_pattern_speeds[index_positive] = (
                delta_phis[index_positive] - np.pi
            ) / age_bin

        self.__bar_pattern_speeds = bar_pattern_speeds

        if self.__info:
            print("Calculation is done!")
        return self.__bar_pattern_speeds

    def calculate_monotone_bar_major_axes(self):
        """
        Shape the major axes' angles into a monotone sequence.
        """
        if self.__info:
            print("The angles will be revised into a monotone sequence.")
        angles = np.array(self.get_bar_major_axes())
        delta_phis = angles[1:] - angles[:-1]
        index_positive = np.where(delta_phis > 0)[0]
        index_negative = np.where(delta_phis <= 0)[0]
        N_positive = len(index_positive)
        N_negative = len(index_negative)
        clockwise = (
            False if N_positive > N_negative else True
        )  # whether clockwise rotation

        for i in range(0, len(angles) - 1):
            if angles[i + 1] < angles[i] and not (clockwise):
                angles[i + 1 :] += np.pi
            elif angles[i + 1] > angles[i] and clockwise:
                angles[i + 1 :] -= np.pi

        self.__bar_major_axes = tuple(angles)
        return angles
