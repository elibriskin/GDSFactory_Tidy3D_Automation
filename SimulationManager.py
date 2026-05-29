import numpy as np
import gdsfactory as gf
from gdsfactory.generic_tech import LAYER_STACK, get_generic_pdk
import gplugins as gp
import gplugins.tidy3d as gt
from gplugins import plot
from gplugins.common.config import PATH
import matplotlib.pyplot as plt
import gdstk
import pprint
import inspect


import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver

class SimulationManager():
    '''
    A manager that can be used to faciliate an automated pipeline for photonic device simulations, specifically
    enabling integration of GDSFactory parametric components (p-cells) with Tidy3D.
    '''

    def __init__(self):
        self.simulation=None
        self.components=[]
        self.current_ports = {}
        self.current_params = {}
        self.reference_plane="bottom",
        self.axis = 2,
        self.slab_bounds=(0, .22),
        self.medium_wg = td.Medium(permittivity=3.48**2),
        self.medium_sub = td.Medium(permittivity=1.45**2)
        self.structure_params = {}
        self.monitor_params = {}
        self.mode_spec_params = {}
        self.mode_params = {}

    def load_component(self, component, component_name, clear_component_data=True):
        '''
        Loads a component into the simulation manager
        '''

        if clear_component_data:
            self.components = []
            self.current_ports = {}
            self.current_params = {}

        #Adds to current component list
        self.components.append(component.copy())

        #Create list of ports associated with current component
        self.current_ports[component_name] = []

        #Extracts all ports from component and appends to current port dictionary
        for port in component.ports:
            self.current_ports[component_name].append(port.name)

        #Extract p-cell function arguments as component parameters
        sig = inspect.signature(getattr(gf.components, component.function_name))

        #Create list of parameters associated with current parameter
        self.current_params[component_name] = list(sig.parameters.keys())

        self.current_component_function = getattr(gf.components, component.function_name)

    def display_current_ports(self):
        '''
        Displays all current ports of device
        '''
        pprint.pprint(self.current_ports)

    def display_component_parameters(self):
        '''
        Displays all current parameters of device
        '''
        pprint.pprint(self.current_params)

    def get_port_coords(self, component, port_name, buffers=(0, 0, 0)):
        '''
        Obtain the coordinates of a given port of a photonic component.
        '''
        return component.ports[port_name].center[0] + buffers[0], component.ports[port_name].center[1]+ buffers[1], 0+ buffers[2]


    def create_port_monitors(self, component, ports, monitor_type, size, freqs, mode_spec=None, buffers=(0,0,0)):
        '''
        Creates a series of monitors situated at the ports of a give photonic component.
        '''
        #List of port monitors
        port_monitors = []

        #Iterate through each specified port
        for port in ports:

            if monitor_type == "flux":

                #Create flux monitor centered at port coordinates
                port_monitor = td.FluxMonitor(
                    center=self.get_port_coords(component, port, buffers=buffers),
                    size=size,
                    freqs=freqs,
                    name=f"{monitor_type}_port_{port}",
                )
            elif monitor_type == "mode":

                #Create mode monitor centered at port coordinates
                port_monitor = td.ModeMonitor(
                    center=self.get_port_coords(component, port, buffers=buffers),
                    size=size,
                    freqs=freqs,
                    mode_spec=mode_spec,
                    name=f"{monitor_type}_port_{port}",
                )
            else:
                raise Exception("Invalid monitor type! Valid monitor types are 'flux' and 'mode")
            
            #Add to list of port monitors
            port_monitors.append(port_monitor)

        return port_monitors


    def convert_to_structure(
            self,
            component, 
            reference_plane="bottom",
            axis = 2,
            slab_bounds=(0, .22),
            medium_wg = td.Medium(permittivity=3.48**2),
            medium_sub = td.Medium(permittivity=1.45**2),
            add_substrate=False,
            substrate_height=.5):
        '''
        Takes in a GDSFactory defined p-cell and converts it into a structure that can be simulated
        in Tidy3D.
        '''

        #Flatten all cells into single p-cell
        component.flatten()

        #Create filename for saving GDS
        filename = f'{list(self.current_params.keys())[0]}.gds'

        #Save file
        component.write_gds(filename)

        #Load in GDS component
        lib_loaded = gdstk.read_gds(filename)

        #Create component geometry
        component_geo = td.Geometry.from_gds(
            lib_loaded.cells[1],
            gds_layer=lib_loaded.cells[1].polygons[0].layer,
            gds_dtype=lib_loaded.cells[1].polygons[0].datatype,
            axis=axis,
            slab_bounds=slab_bounds,
            reference_plane=reference_plane

        )

        #Convert component geometry to structure
        component_structure = td.Structure(geometry=component_geo, medium=medium_wg)

        #Return discrete p-cell structure without substrate
        if add_substrate == False:
            return component_structure
        else:
            
            #Create a substrate the size of bounding box of p-cell
            substrate = td.Structure(
                geometry = td.Box(
                    center=(component_geo.bounding_box.center[0], component_geo.bounding_box.center[1], -slab_bounds[1]/2),
                    size=(component_geo.bounding_box.size)
                ),
                medium=medium_sub,
                name="Substrate"
            )

            #Return both component structure and substrate structure
            structures = [substrate, component_structure]
            return structures
        
    def create_structures(self,
            reference_plane="bottom",
            axis = 2,
            slab_bounds=(0, .22),
            medium_wg = td.Medium(permittivity=3.48**2),
            medium_sub = td.Medium(permittivity=1.45**2),
            add_substrate=False,
            substrate_height=.5):
            '''
            This method takes the components loaded into the simulation manager and converts them into
            structures to be used in a simulation.
            '''

            #Create list of structures
            structures = []

            for component in self.components:
                structure = self.convert_to_structure(
                    component=component,
                    reference_plane=reference_plane,
                    axis=axis,
                    slab_bounds=slab_bounds,
                    medium_wg=medium_wg,
                    medium_sub=medium_sub,
                    add_substrate=add_substrate,
                    substrate_height=substrate_height
                )

                if add_substrate:
                    for s in structure:
                        structures.append(s)
                else:        
                    structures.append(structure)
            
            return structures

    def setup_simulation(self,
            grid_spec, 
            run_time,
            boundary_spec,
            monitor_size=0.1,
            mode_size=0.1,
            sim_buffers=(0, 0, 0)):
        '''
        Sets up a simulation from loaded parameters.
        '''
        
        #Create structures from components
        structures = self.create_structures(**self.structure_params)

        #Setup simulation size
        sim_size = list(structures[0].geometry.bounding_box.size)
        sim_center = list(structures[0].geometry.bounding_box.center)

        #Add buffers to simulation size
        sim_size[0] += sim_buffers[0]
        sim_size[1] += sim_buffers[1]
        sim_size[2] += sim_buffers[2]

        #Modify size of monitors and mode source
        self.monitor_params["size"] = (0, sim_size[1]*monitor_size, td.inf)
        self.mode_params["size"] = (0, sim_size[1]*mode_size, td.inf)
        
        #Create monitors located at ports
        monitors = self.create_port_monitors(component=self.components[0], **self.monitor_params)
        
        #Create mode specification
        mode_spec = td.ModeSpec(**self.mode_spec_params)

        #Create mode source
        mode_source = [td.ModeSource(mode_spec=mode_spec, **self.mode_params)]

        #Create simulation object
        sim = td.Simulation(
            size=sim_size,
            center=sim_center,
            grid_spec=grid_spec,
            structures=structures,
            monitors=monitors,
            sources=mode_source,
            run_time=run_time,
            boundary_spec=boundary_spec,
        )

        return sim
    
    def create_simulation_sweep(self,
            parameter,
            parameter_list,
            grid_spec, 
            run_time,
            boundary_spec,
            monitor_size=0.1,
            mode_size=0.1,
            sim_buffers=(0, 0, 0)):
        
        '''
        Sets up a list of simulations corresponding to a parameterized sweep of a given component.
        '''
        
        #Instantiate list of simulations
        simulation_list = []
        
        #Iterate through each parametevaluesr in array of parameter values
        for parameter_value in parameter_list:

            #Create component arguments
            param_data = {parameter: parameter_value}

            #Create parameterized component
            component = self.current_component_function(**param_data)

            #Load component into simulation manaager
            self.load_component(component, component.function_name)

            #Create structures from components
            structures = self.create_structures(**self.structure_params)

            #Setup simulation size
            sim_size = list(structures[0].geometry.bounding_box.size)
            sim_center = list(structures[0].geometry.bounding_box.center)

            #Add buffers to simulation size
            sim_size[0] += sim_buffers[0]
            sim_size[1] += sim_buffers[1]
            sim_size[2] += sim_buffers[2]

            #Modify size of monitors and mode source
            self.monitor_params["size"] = (0, sim_size[1]*monitor_size, td.inf)
            self.mode_params["size"] = (0, sim_size[1]*mode_size, td.inf)
            
            #Create monitors located at ports
            monitors = self.create_port_monitors(component=self.components[0], **self.monitor_params)
            
            #Create mode specification
            mode_spec = td.ModeSpec(**self.mode_spec_params)

            #Create mode source
            mode_source = [td.ModeSource(mode_spec=mode_spec, **self.mode_params)]

            #Create simulation with parameterized component
            sim = td.Simulation(
                size=sim_size,
                center=sim_center,
                grid_spec=grid_spec,
                structures=structures,
                monitors=monitors,
                sources=mode_source,
                run_time=run_time,
                boundary_spec=boundary_spec,
            )

            #Add to simulation sweep list
            simulation_list.append(sim)

        return simulation_list
      


