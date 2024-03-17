import numpy as np
from PIL import Image, ImageEnhance

# Helper functions
def normalize(v):
    return v / np.linalg.norm(v)
def reflect(I, N):
    return I - 2 * np.dot(I, N) * N
class Material:
    def __init__(self, name):
        self.name = name
        self.diffuse_color = [1.0, 1.0, 1.0]  # Default white
        self.ambient_color = [1.0, 1.0, 1.0]  # Default white for ambient color
        self.specular_color = [1.0, 1.0, 1.0]  # Default white for specular color
        self.shininess = 32.0  # Default shininess (specular highlight size)
class OBJLoader:
    def __init__(self, obj_filename, mtl_filename=None):
        self.vertices = []
        self.faces = []
        self.materials = {}
        self.current_material= None
        if mtl_filename:
            self.load_mtl(mtl_filename)
            self.load_obj(obj_filename)

    def load_mtl(self, filename):
        with open(filename, 'r') as file:
            current_material = None
            for line in file:
                if line.startswith('newmtl '):
                    current_material = Material(line.split()[1].strip())
                    self.materials[current_material.name] = current_material
                elif line.startswith('Kd ') and current_material:
                    current_material.diffuse_color = list(map(float, line.split()[1:4]))
                elif line.startswith('Ka ') and current_material:
                    current_material.ambient_color = list(map(float, line.split()[1:4]))
                elif line.startswith('Ks ') and current_material:
                    current_material.specular_color = list(map(float, line.split()[1:4]))
                elif line.startswith('Ns ') and current_material:
                    current_material.shininess = float(line.split()[1])

    def load_obj(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    self.vertices.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith('usemtl '):
                    self.current_material = line.split()[1].strip()
                elif line.startswith('f '):
                    face_data = line.strip().split()[1:4]
                    face_vertices = [int(face.split('/')[0]) for face in face_data]
                    face_material = self.current_material
                    self.faces.append((face_vertices, face_material))

    def get_triangles(self):
        for face, material_name in self.faces:
            material = self.materials.get(material_name, None)
            yield [self.vertices[idx - 1] for idx in face], material
            # Example usage
            # obj_loader = OBJLoader('path_to_your_obj_file.obj', 'path_to_your_mtl_file.mtl')
            # for triangle, material in obj_loader.get_triangles():
            # Use material.diffuse_color and other properties in your rendering calculations

    
class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)
        self.intensity = float(intensity)
class Camera:
    def __init__(self, position, look_at, up_vector, fov, width, height):
        self.position = np.array(position, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.up_vector = np.array(up_vector, dtype=float)
        self.fov = float(fov)
        self.width = int(width)
        self.height = int(height)
        self.aspect_ratio = float(width) / height
        self.initialize_camera()

    def initialize_camera(self):
        self.direction = (self.look_at - self.position)
        self.direction /= np.linalg.norm(self.direction)
        self.right_vector = np.cross(self.direction, self.up_vector)
        self.right_vector /= np.linalg.norm(self.right_vector)
        self.up_vector = np.cross(self.right_vector, self.direction)
        self.half_width = np.tan(np.radians(self.fov / 2))
        self.half_height = self.half_width / self.aspect_ratio

    def get_ray(self, x, y):
        x_comp = self.right_vector * ((x / float(self.width) - 0.5) * self.half_width)
        y_comp = self.up_vector * ((y / float(self.height) - 0.5) * self.half_height)
        ray_direction = self.direction + x_comp + y_comp
        return self.position, ray_direction / np.linalg.norm(ray_direction)
    # Example camera setup
    # camera = Camera(position=[0, 0, 5], look_at=[0, 0, 0], up_vector=[0, 1, 0], fov=90, width=800, height=600)

class AABB:
    def __init__(self, min_point, max_point):
        self.min = np.array(min_point, dtype=float)
        self.max = np.array(max_point, dtype=float)

    def intersect(self, ray_origin, ray_direction):
        # Implementation of the slab method
        t_min = (self.min - ray_origin) / ray_direction
        t_max = (self.max - ray_origin) / ray_direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return False

        return True
def ray_triangle_intersect(ray_origin, ray_direction, triangle):
    vertex0, vertex1, vertex2 = triangle
    edge1, edge2 = np.subtract(vertex1, vertex0), np.subtract(vertex2, vertex0)
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -1e-7 < a < 1e-7:
        return False, None
    f = 1.0 / a
    s = np.subtract(ray_origin, vertex0)
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > 1e-7:
        intersection_point = ray_origin + ray_direction * t
        return True, intersection_point
    else:
        return False, None
        # Example usage
        # intersection, point = ray_triangle_intersect(ray_origin, ray_direction, triangle)
def calculate_lighting(intersection_point, normal, material, lights, camera_position):
    ambient_light = np.array([0.2, 0.2, 0.2])  # Global ambient light
    color = np.array(material.ambient_color) * ambient_light  # Initial color is ambient

    for light in lights:
        # Calculate diffuse lighting
        light_dir = normalize(light.position - intersection_point)
        diffuse_intensity = max(np.dot(light_dir, normal), 0)
        color += np.array(material.diffuse_color) * diffuse_intensity * np.array(light.color)

        # Calculate specular lighting
        view_dir = normalize(camera_position - intersection_point)
        reflect_dir = reflect(-light_dir, normal)
        specular_intensity = np.power(max(np.dot(view_dir, reflect_dir), 0), material.shininess)
        color += np.array(material.specular_color) * specular_intensity * np.array(light.color)
        print(f"Material Color: {material.diffuse_color}")
        print(f"Light Color: {light.color}")
        print(f"Calculated Color: {color}")

    return np.clip(color, 0, 1)
    # Example usage in the ray trace loop:
    # ...
    # if hit:
    #     normal = compute_normal_at_intersection(...)  # Compute the normal at the intersection
    #     color = calculate_lighting(point, normal, material, lights, camera.position)
    # ...
def ray_trace(ray_origin, ray_direction, obj_loader, lights, camera_position, depth=0, max_depth=3):
    if depth > max_depth:
        return np.array([0, 0, 0])  # Base case for recursion (e.g., for reflections)

    closest_hit = None
    closest_material = None
    min_distance = np.inf

    # Test for intersections with all triangles
    for triangle_vertices, material in obj_loader.get_triangles():
        # Convert triangle vertices to NumPy arrays
        triangle = [np.array(vertex) for vertex in triangle_vertices]

        hit, hit_point = ray_triangle_intersect(ray_origin, ray_direction, triangle)
        if hit:

            distance = np.linalg.norm(hit_point - ray_origin)


            if distance < min_distance:
                min_distance = distance
                closest_hit = hit_point
                closest_material = material

    # Calculate color based on closest hit
    if closest_hit is not None and closest_material is not None:
        # Compute normal, assume triangles are flat surfaces
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal = normalize(normal)  # Normalize the normal vector
        return calculate_lighting(closest_hit, normal, closest_material, lights, camera_position)
    else:
        return np.array([0, 0, 0])  # Background color or environment map color
        # Replace 'calculate_lighting' with your lighting calculation function
        # Example usage
        # image = ray_trace(camera, obj_loader)
def create_test_scene(obj_loader, camera, lights):
    image = Image.new('RGB', (camera.width, camera.height))
    pixels = image.load()

    for y in range(camera.height):
        for x in range(camera.width):
            ray_origin, ray_direction = camera.get_ray(x, y)
            color = ray_trace(ray_origin, ray_direction, obj_loader, lights, camera.position)
            pixels[x, y] = tuple((255 * np.clip(color, 0, 1)).astype(int))

    image.save('render.png')
    enhancer = ImageEnhance.Brightness(image)
    brighter_image = enhancer.enhance(1.5)  # Increase brightness by 50%
    brighter_image.save('rendered_brighter.png')
if __name__ == "__main__":
    obj_file = r"C:\Users\micha\dev\OpenGL\triangle.obj"
    mtl_file = r"C:\Users\micha\dev\OpenGL\triangle.mtl"

    obj_loader = OBJLoader(obj_file, mtl_file)
    camera = Camera(position=[0, 7, 0], look_at=[0, 0, 0], up_vector=[0, 0, 1], fov=90, width=800, height=600)
    lights = [
        Light(position=[1, 3, 2], color=[1, 1, 1], intensity=5.0),  # A white light

    ]


    create_test_scene(obj_loader,camera,lights)