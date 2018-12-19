/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
    unsigned int pos = 0;
    unsigned int val;

    //Something should happen here

    int id = get_global_id(0);
    data[id] = max(data[id * 2], data[id * 2 + 1]);
}
