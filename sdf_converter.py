import numpy as np

filename = 'formation4_enlarging'
description = 'This SDF file is for enlarging environment'


class Wall:
    def __init__(self, vertex1, vertex2):
        self.__vertex1 = np.array(vertex1[:2])
        self.__vertex2 = np.array(vertex2[:2])

    @property
    def center(self):
        return (self.__vertex1 + self.__vertex2) / 2

    @property
    def length(self):
        return np.linalg.norm(self.__vertex2 - self.__vertex1)

    @property
    def angle(self):
        return np.arctan2(self.__vertex2[1] - self.__vertex1[1],
                          self.__vertex2[0] - self.__vertex1[0])


with open(f'scenarios_unicycle/scenarios/{filename}.yml', 'r') as file:
    import yaml

    scenario, _, _ = yaml.safe_load(file).values()

# Set the static obstacles
walls = [Wall(o[idx], o[idx + 1]) for o in scenario['obstacles'] for idx in range(len(o) - 1)]

with open(f'scenarios_unicycle/scenarios/model.sdf', 'w') as file:
    obstacles = '\n'.join([f'''
    <collision name='obs_wall_{idx + 1}'>
        <pose>{w.center[0]} {w.center[1]} 0.25 0 0 {w.angle:.3f}</pose>
        <geometry>
            <box><size>{w.length:.3f} 0.01 0.5</size></box>
        </geometry>
        <max_contacts>10</max_contacts>
        <surface><bounce/><friction><ode/></friction><contact><ode/></contact></surface>
    </collision>

    <visual name='obs_wall_{idx + 1}'>
        <pose>{w.center[0]} {w.center[1]} 0.25 0 0 {w.angle:.3f}</pose>
        <geometry>
            <box><size>{w.length:.3f} 0.01 0.5</size></box>
        </geometry>
        <material>
            <script>
                <uri>file://media/materials/scripts/gazebo.material</uri>
                <name>Gazebo/Bricks</name>
            </script>
        </material>
    </visual>
    ''' for idx, w in enumerate(walls)])

    file.write(f'''
    <sdf version='1.5'>
        <!-- This SDF file is for enlarging environment -->
        <model name='ros_symbol'>
            <static>1</static>
            <link name='symbol'>
                {obstacles}
            </link>
      </model>
    </sdf>
    ''')
