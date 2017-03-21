#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#define GLM_FORCE_RADIANS

using namespace std;

GLuint window_id;
bool preview_mode = true;

struct
{
    glm::mat4 model_matrix;
    glm::mat4 projection_matrix;
    glm::mat4 view_matrix;
    glm::vec4 viewport;
} matrices;

struct Model
{
    glm::vec3 position;
    glm::mat4 rotation;
    float scaling;

    struct
    {
        vector<glm::vec3> vertex_positions;
        vector<glm::vec3> vertex_normals;
        vector<glm::vec3> faces; // TODO: vec3 does not make sense
    } mesh;

    struct
    {
        glm::vec3 ambient;
        glm::vec3 diffuse;
        glm::vec3 specular;
        float shininess;
    } material;
};

vector<Model> models;

struct Ray
{
    glm::vec3 start;
    glm::vec3 end;
    glm::vec3 direction;
    glm::vec2 screen_position;
};

struct TriangleIntersection
{
    glm::vec3 position;
    glm::vec3 face;
    Model *model; // TODO: needed?
    glm::vec3 color;
};

struct {
    vector<Ray> rays;
    vector<TriangleIntersection> triangle_intersections;
} raytrace;

struct Lightsource
{
    glm::vec3 position;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
};

vector<Lightsource> light_sources;

glm::mat4 rotation = glm::mat4(1);

bool load_mesh(const string &path, bool clockwise_polygon_winding, Model &model)
{
    ifstream file(path);
    string line;

    model.mesh.vertex_positions.clear();
    model.mesh.vertex_normals.clear();
    model.mesh.faces.clear();

    // read the first line (should contain the "OFF" statement)
    getline(file, line);
    if (line.compare("OFF") != 0)
    {
        cerr << "File not having the correct format!" << endl;
        return false;
    }

    // read the next line containing v: vertices, f: faces, e: edges
    int v, f, e;
    getline(file, line);
    std::istringstream stream1(line);
    if (!(stream1 >> v >> f >> e))
    {
        cerr << "Could not correctly read out information!" << endl;
        return false;
    }

    // get the vertex positions
    float x, y, z;
    for (int i = 0; i < v; i++)
    {
        getline(file, line);
        std::istringstream stream2(line);
        stream2 >> x >> y >> z;
        model.mesh.vertex_positions.push_back(glm::vec3(x, y, z));
    }

    // get the faces
    int number_vertices, i1, i2, i3;
    for (int i = 0; i < f; i++)
    {
        getline(file, line);
        std::istringstream stream2(line);
        stream2 >> number_vertices;
        if (number_vertices != 3)
        {
            cerr << "Don't know what to do with more or less than 3!" << endl;
            cerr << number_vertices << " " << f << " " << i << endl;
            return false;
        }
        stream2 >> i1 >> i2 >> i3;

        // TODO: verify
        if (clockwise_polygon_winding)
            model.mesh.faces.push_back(glm::vec3(i1, i2, i3));
        else
            model.mesh.faces.push_back(glm::vec3(i1, i3, i2));
    }

    // calculate vertex normals
    for (int i = 0; i < model.mesh.vertex_positions.size(); i++)
        model.mesh.vertex_normals.push_back(glm::vec3(0.0, 0.0, 0.0));

    for (int i = 0; i < model.mesh.faces.size(); i++){
        glm::vec3 pos_1 = model.mesh.vertex_positions[model.mesh.faces[i].x];
        glm::vec3 pos_2 = model.mesh.vertex_positions[model.mesh.faces[i].y];
        glm::vec3 pos_3 = model.mesh.vertex_positions[model.mesh.faces[i].z];

        // get two vectors between points
        glm::vec3 u = pos_2 - pos_1;
        glm::vec3 v = pos_3 - pos_1;

        // calc cross product and normalize
        glm::vec3 normal = glm::normalize(glm::cross(u, v));

        // For our three vertices add the normal to it
        model.mesh.vertex_normals[model.mesh.faces[i].x] += normal;
        model.mesh.vertex_normals[model.mesh.faces[i].y] += normal;
        model.mesh.vertex_normals[model.mesh.faces[i].z] += normal;
    }

    for (int i = 0; i < model.mesh.vertex_positions.size(); i++)
        model.mesh.vertex_normals[i] = glm::normalize(model.mesh.vertex_normals[i]);

    return true;
}

glm::mat4 calculate_model_matrix(Model *model)
{
    glm::mat4 modelMatrix = glm::translate(glm::vec3(model->position.x, model->position.y, model->position.z))
                            * model->rotation
                            * scale(glm::vec3(model->scaling));

    return modelMatrix;
}

glm::mat4 calculate_normal_matrix(Model *model)
{
    return glm::transpose(glm::inverse(calculate_model_matrix(model)));
}

void render_preview(void)
{
    // render our model meshes
    for (Model &model: models)
    {
        for (glm::vec3 &face: model.mesh.faces)
        {
            // apply model matrix for translation/rotation/scaling
            glm::mat4 model_matrix = calculate_model_matrix(&model);
            glPushMatrix();
            glMultMatrixf(&model_matrix[0][0]);

            // draw the model's mesh
            glBegin(GL_TRIANGLES);
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(model.mesh.vertex_positions[face.x].x, model.mesh.vertex_positions[face.x].y, model.mesh.vertex_positions[face.x].z);
            glVertex3f(model.mesh.vertex_positions[face.y].x, model.mesh.vertex_positions[face.y].y, model.mesh.vertex_positions[face.y].z);
            glVertex3f(model.mesh.vertex_positions[face.z].x, model.mesh.vertex_positions[face.z].y, model.mesh.vertex_positions[face.z].z);
            glEnd();

            // revert the model matrix
            glPopMatrix();
        }
    }

    // render light sources
    for (Lightsource &light_source: light_sources)
    {
        glPushMatrix();
        glTranslatef(light_source.position.x, light_source.position.y, light_source.position.z);
        glColor3f(1.0, 1.0, 0.0);
        glutSolidSphere(0.05f, 20, 20);
        glPopMatrix();
    }
}

void render_raytrace(void)
{
    for (TriangleIntersection &intersection: raytrace.triangle_intersections)
    {
        glBegin(GL_POINTS);
        glColor3f(intersection.color.x, intersection.color.y, intersection.color.z);
        glVertex3f(intersection.position.x, intersection.position.y, intersection.position.z);
        glEnd();
    }
}

// stolen from:
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
int triangle_intersection(const glm::vec3 &V1,
                          const glm::vec3 &V2,
                          const glm::vec3 &V3,
                          const glm::vec3 &O,
                          const glm::vec3 &D,
                          float* out)
{
    glm::vec3 e1, e2;
    glm::vec3 P, Q, T;
    float det, inv_det, u, v;
    float t;

    const float epsilon = 0.000001;

    // find vectors for two edges sharing V1
    e1 = V2 - V1;
    e2 = V3 - V1;

    // begin calculating determinant - also used to calculate u parameter
    P = glm::cross(D, e2);

    // if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
    det = glm::dot(e1, P);
    if (det > -epsilon && det < epsilon)
        return 0;

    // calculate invert determinant
    inv_det = 1.f / det;

    // calculate distance from V1 to ray origin
    T = O - V1;

    // calculate u parameter and test bound
    // and abort if the intersection lies outside of the triangle
    u = glm::dot(T, P) * inv_det;
    if (u < 0.f || u > 1.f)
        return 0;

    // prepare to test v parameter
    Q = glm::cross(T, e1);

    // calculate V parameter and test bound
    v = glm::dot(D, Q) * inv_det;

    // the intersection lies outside of the triangle
    if (v < 0.f || u + v  > 1.f)
        return 0;

    // now check again if we've found an intersection and calculate the result
    t = glm::dot(e2, Q) * inv_det;
    if (t > epsilon)
    {
        *out = t;
        return 1;
    }

    // no hit, no win
    return 0;
}

float blinnPhongReflection(glm::vec3 lightSource, glm::vec3 position, glm::vec3 normal)
{
  glm::vec3 E = position;
  glm::vec3 L = position - lightSource;

  glm::vec3 bisector = glm::normalize(E + L);

  float light = max(0.0f, glm::dot(bisector, normal)); // TODO: normal wrong direction?

  return light;
}

void generate_raytrace(void)
{
    // in here you will encounter some nasty conversion between vec3 and vec4.
    // this could and should be simplified by using homogenous vec4 coordinates throughout
    // the entire project.
    // also vertices and normals should be multiplied with the model/normal matrix only once.
    // the current implementation causes a big performance regression.

    // clear previous output
    raytrace.rays.clear();
    raytrace.triangle_intersections.clear();

    // implementation of "createPrimaryRays()"
    cout << "generate primary rays" << endl;
    for (float x = 0.0f; x < matrices.viewport[2]; x += 1.0f)
    {
        for (float y = 0.0f; y < matrices.viewport[3]; y += 1.0f)
        {
            glm::vec3 ray_start = glm::unProject(glm::vec3(x, y, 0.0f), matrices.view_matrix, matrices.projection_matrix, matrices.viewport);
            glm::vec3 ray_end = glm::unProject(glm::vec3(x, y, 1.0f), matrices.view_matrix, matrices.projection_matrix, matrices.viewport);

            // save ray to our raytrace structure
            Ray new_ray;
            new_ray.start = ray_start;
            new_ray.end = ray_end;
            new_ray.direction = glm::normalize(ray_end - ray_start);
            new_ray.screen_position = glm::vec2(x, y);
            raytrace.rays.push_back(new_ray);
        }
    }

    // calculate mesh intersection for each generated ray
    cout << "calculate triangle intersection for each ray" << endl;
    for (Ray &ray: raytrace.rays)
    {
        float intersection_t = FLT_MAX;
        TriangleIntersection intersection;

        for (Model &model: models)
        {
            glm::mat4 mm = calculate_model_matrix(&model);

            for (unsigned int i = 0; i < model.mesh.faces.size(); i++)
            {
                glm::vec3 &face = model.mesh.faces[i];

                glm::vec3 v0 = glm::vec3(mm * glm::vec4(model.mesh.vertex_positions[face.x].x, model.mesh.vertex_positions[face.x].y, model.mesh.vertex_positions[face.x].z, 1.0f));
                glm::vec3 v1 = glm::vec3(mm * glm::vec4(model.mesh.vertex_positions[face.y].x, model.mesh.vertex_positions[face.y].y, model.mesh.vertex_positions[face.y].z, 1.0f));
                glm::vec3 v2 = glm::vec3(mm * glm::vec4(model.mesh.vertex_positions[face.z].x, model.mesh.vertex_positions[face.z].y, model.mesh.vertex_positions[face.z].z, 1.0f));

                // there might be multiple intersections between our ray and the triangles from the mesh.
                // we only want to save the one closest to the ray origin.
                float t;
                if (triangle_intersection(v0, v1, v2, ray.start, ray.direction, &t) && t < intersection_t)
                {
                    intersection_t = t;
                    intersection.position = ray.start + t * ray.direction,
                    intersection.face     = face;
                    intersection.model    = &model;
                }
            }
        }

        if (intersection_t < FLT_MAX)
            raytrace.triangle_intersections.push_back(intersection);
    }

    // TODO
    cout << "calculate pixel color with lighting foo bar" << endl;
    for (TriangleIntersection &intersection: raytrace.triangle_intersections)
    {
        Model *model = intersection.model;

        glm::mat4 nm = calculate_normal_matrix(model);

        glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);

        // TODO
        glm::vec3 face_normal = model->mesh.vertex_normals[intersection.face.x]
                              + model->mesh.vertex_normals[intersection.face.y]
                              + model->mesh.vertex_normals[intersection.face.z];
        face_normal = glm::vec3(nm * glm::vec4(face_normal, 1.0f));
        face_normal /= 3.0f;

        for (Lightsource &light_source: light_sources)
        {
            // ambient part
            color += light_source.ambient * model->material.ambient;

            // diffuse part
            color += light_source.diffuse * model->material.diffuse
                     * max(0.0f, glm::dot(glm::normalize(intersection.position - light_source.position), -face_normal));

            // specular part
            color += light_source.specular * model->material.specular
                     * pow(blinnPhongReflection(light_source.position, intersection.position, -face_normal), model->material.shininess);
        }

        intersection.color = color;
    }

    cout << "ray tracer finished" << endl;
}

void render(void)
{
    // clear the image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // load the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&matrices.projection_matrix[0][0]);

    // load the view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(&matrices.view_matrix[0][0]);

    // TODO:
    glMultMatrixf(&rotation[0][0]);

    if (preview_mode)
        render_preview();
    else
        render_raytrace();

    // swap front and back buffer
    glutSwapBuffers();
}

void reshape(int width, int height)
{
    // higher value means that camera is farther away from object
    float camera_z = 10.0;

    // TODO: what are we doing here?
    matrices.viewport = glm::vec4(0.0f, 0.0f, (float)width, (float)height);
    glViewport(0, 0, width, height);

    // recalculate projection matrix
    matrices.projection_matrix = glm::perspective(glm::radians(45.0), (double)width / (double)height,
                                                  camera_z / 10.0, camera_z * 10.0);

    // recalculate view matrix
    matrices.view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, camera_z), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
}

void keypress(unsigned char key, int x, int y)
{
    glm::vec2 v;

    switch(key)
    {
        case ' ': preview_mode = !preview_mode; break;
        case 'r': generate_raytrace(); break;

        case 'n':
            v = glm::vec2(-0.05f, 0.0f);
            rotation *= glm::rotate(glm::mat4(1),
                                    glm::radians(180 * glm::length(v)),
                                    glm::normalize(glm::vec3(v.y, v.x, 0.0f)));
        break;
        case 'm':
            v = glm::vec2(+0.05f, 0.0f);
            rotation *= glm::rotate(glm::mat4(1),
                                    glm::radians(180 * glm::length(v)),
                                    glm::normalize(glm::vec3(v.y, v.x, 0.0f)));
        break;
    }

    glutPostRedisplay();
}

int main(int argc, char *argv[])
{
    // init glut
    glutInit(&argc, argv);
    glutInitContextVersion(2, 1); // TODO: Linux only?
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutSetOption(GLUT_RENDERING_CONTEXT, GLUT_USE_CURRENT_CONTEXT);

    // create a window and set up the callbacks
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(800, 600);
    window_id = glutCreateWindow("CG1 Ray Tracer");
    glutSetWindow(window_id);
    glutDisplayFunc(render);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keypress);

    // init glew
    glewExperimental = GL_TRUE;
    unsigned int glew_result = glewInit();
    if (glew_result != GLEW_OK)
    {
        cerr << "GLEW initialization failed: " << glew_result << endl;
        return 1;
    }

    // some debug prints
    cout << "GPU: " << glGetString(GL_RENDERER) << endl
         << "OpenGL version: " << glGetString(GL_VERSION) << endl
         << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl
         << "Geometry shaders supported: " << (glewIsSupported("GL_EXT_geometry_shader4") ? "yes" : "no") << endl
         << endl;

    // teapot
    Model teapot;

    teapot.position = glm::vec3(0.0f, -1.0f, 0.0f);
    teapot.rotation = glm::rotate(glm::mat4(1),
                                  glm::radians(40.0f),
                                  glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)));
    teapot.scaling  = 1.0f;

    teapot.material.ambient   = glm::vec3(1.0f, 1.0f, 1.0f);
    teapot.material.diffuse   = glm::vec3(1.0f, 1.0f, 1.0f);
    teapot.material.specular  = glm::vec3(1.0f, 1.0f, 1.0f);
    teapot.material.shininess = 10.0f;

    // TODO: adjust path
    if (!load_mesh("./meshes/teapot.off", true, teapot))
    {
        cerr << "Could not load teapot mesh." << endl;
        return 1;
    }

    models.push_back(teapot);

    // back wall
    Model back_wall;

    back_wall.position = glm::vec3(0.0f, 0.0f, -5.0f);
    back_wall.rotation = glm::rotate(glm::mat4(1),
                                     glm::radians(0.0f),
                                     glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)));
    back_wall.scaling = 10.0f;

    back_wall.material.ambient   = glm::vec3(1.0f, 1.0f, 1.0f);
    back_wall.material.diffuse   = glm::vec3(1.0f, 1.0f, 1.0f);
    back_wall.material.specular  = glm::vec3(1.0f, 1.0f, 1.0f);
    back_wall.material.shininess = 10.0f;

    // TODO: adjust path
    if (!load_mesh("./meshes/plane4x4.off", true, back_wall))
    {
        cerr << "Could not load plane4x4 mesh." << endl;
        return 1;
    }

    models.push_back(back_wall);

    // floor
    Model floor;

    floor.position = glm::vec3(0.0f, -1.0f, 0.0f);
    floor.rotation = glm::rotate(glm::mat4(1),
                                     glm::radians(90.0f),
                                     glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)));
    floor.scaling = 10.0f;

    floor.material.ambient   = glm::vec3(1.0f, 1.0f, 1.0f);
    floor.material.diffuse   = glm::vec3(1.0f, 1.0f, 1.0f);
    floor.material.specular  = glm::vec3(1.0f, 1.0f, 1.0f);
    floor.material.shininess = 10.0f;

    // TODO: adjust path
    if (!load_mesh("./meshes/plane4x4.off", true, floor))
    {
        cerr << "Could not load plane4x4 mesh." << endl;
        return 1;
    }

    models.push_back(floor);

    // initialize light sources
    light_sources.push_back({
        .position = glm::vec3(0.0f, 0.0f, 0.0f),
        .ambient  = glm::vec3(0.1f, 0.1f, 0.1f),
        .diffuse  = glm::vec3(0.0f, 0.0f, 0.0f),
        .specular = glm::vec3(0.0f, 0.0f, 0.0f)
    });
    light_sources.push_back({
        .position = glm::vec3(3.0f, 3.0f, 3.0f),
        .ambient  = glm::vec3(0.0f, 0.0f, 0.0f),
        .diffuse  = glm::vec3(0.0f, 0.25f, 0.5f),
        .specular = glm::vec3(0.1f, 0.1f, 0.1f)
    });
    light_sources.push_back({
        .position = glm::vec3(-3.0f, -3.0f, -3.0f),
        .ambient  = glm::vec3(0.0f, 0.0f, 0.0f),
        .diffuse  = glm::vec3(0.5f, 0.0f, 0.0f),
        .specular = glm::vec3(0.1f, 0.1f, 0.1f)
    });

    // run the glut event loop
    glutMainLoop();

    return 0;
}
