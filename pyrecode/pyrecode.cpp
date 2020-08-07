#include <Python.h>
#include "structmember.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "reader.h"

typedef struct {
	PyObject_HEAD
	uint16_t ny;
	uint16_t nx;
	uint8_t bit_depth;
	uint64_t pow2_lookup_table[64];
} RecodeReader;

static PyMemberDef ReCoDe_members[] = {
	{ "ny", T_USHORT, offsetof(RecodeReader, ny), 0, "rows in frame" },
	{ "nx", T_USHORT, offsetof(RecodeReader, nx), 0, "cols in frame" },
	{ "bit_depth", T_UBYTE, offsetof(RecodeReader, bit_depth), 0, "bits per pixel" },
	{ "pow2_lookup_table", T_ULONG, offsetof(RecodeReader, pow2_lookup_table), 0, "look-up table for bit unpacking" },
	{ NULL }  /* Sentinel */
};

static void 
ReCoDe_dealloc(RecodeReader *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
ReCoDe_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	RecodeReader *self;
	self = (RecodeReader *)type->tp_alloc(type, 0);
	if (self != NULL) {
		self->ny = 0;
		self->nx = 0;
		self->bit_depth = 0;
		for (uint8_t t = 0; t < 64; t++) {
			self->pow2_lookup_table[t] = pow(2, t);
		}
	}
	return (PyObject *)self;
}

static PyObject *
create_buffers (RecodeReader* self, PyObject* args) {

	uint16_t ny, nx;
	uint8_t bit_depth;
	if (!PyArg_ParseTuple(args, "HHb", &ny, &nx, &bit_depth)) {
		printf("Failed to parse arguments in RecodeReader constructor. Expected three: ny, nx, bit_depth\n");
		return Py_BuildValue("i", 0);
	}
	if (self != NULL) {
		self->ny = ny;
		self->nx = nx;
		self->bit_depth = bit_depth;
	}
	return Py_BuildValue("i", 1);
}

PyObject *
bit_unpack_pixel_intensities (RecodeReader *self, PyObject* args) {

	Py_buffer view_frameData;
	Py_buffer deCompressedPixvals;
    uint64_t n_values;

	if (!PyArg_ParseTuple(args, "Ky*y*", &n_values, &deCompressedPixvals, &view_frameData)) {
		return Py_BuildValue("s", "Unable to parse argument frame_index");
		return Py_BuildValue("k", 0);
	}

	int64_t n = _bit_unpack_pixel_intensities (
		n_values, self->bit_depth,
		(uint8_t *)(&deCompressedPixvals)->buf,
		(uint64_t *)(&view_frameData)->buf
	);

	return Py_BuildValue("L", n);
}

PyObject *
get_frame_sparse (RecodeReader *self, PyObject* args) {

	Py_buffer view_frameData;
	Py_buffer deCompressedBinaryImage;
	Py_buffer deCompressedPixvals;
    uint8_t reduction_level;

	if (!PyArg_ParseTuple(args, "by*y*y*", &reduction_level, &deCompressedBinaryImage, &deCompressedPixvals, &view_frameData)) {
		return Py_BuildValue("s", "Unable to parse argument frame_index");
		return Py_BuildValue("k", 0);
	}
	/*
	printf("%d, %d, %d\n", n_compressed_bytes_in_binary_image, n_compressed_bytes_in_pixvals, n_bytes_in_packed_pixvals);
	printf("current position (pyrecode.cpp): %d\n", ftell(self->file));
	*/
	int64_t n = _unpack_frame_sparse (
		self->nx, self->ny, self->bit_depth,
		(uint8_t *)(&deCompressedBinaryImage)->buf, (uint8_t *)(&deCompressedPixvals)->buf,
		(uint64_t *)(&view_frameData)->buf,
		reduction_level
	);
	//printf("Decoded Frame with %d foreground pixels\n", n);
	return Py_BuildValue("L", n);
}

PyObject *
bit_pack_pixel_intensities (RecodeReader *self, PyObject* args) {

	uint64_t sz_packedPixval;
	uint32_t n_fg_pixels;
	uint8_t bit_depth;
	Py_buffer view_pixvals;
	Py_buffer view_packed_pixvals;
	
	if (!PyArg_ParseTuple(args, "Kkby*y*", &sz_packedPixval, &n_fg_pixels, &bit_depth, &view_pixvals, &view_packed_pixvals)) {
		return Py_BuildValue("s", "Unable to parse argument frame_index");
		return Py_BuildValue("k", 0);
	}
	/*
	printf("%d, %d, %d\n", n_compressed_bytes_in_binary_image, n_compressed_bytes_in_pixvals, n_bytes_in_packed_pixvals);
	printf("current position (pyrecode.cpp): %d\n", ftell(self->file));
	*/
	float t = _bit_pack_pixel_intensities (sz_packedPixval, n_fg_pixels, bit_depth, (uint16_t *)(&view_pixvals)->buf, (uint8_t *)(&view_packed_pixvals)->buf);
	//printf("Decoded Frame with %d foreground pixels\n", n);
	return Py_BuildValue("f", t);
}

PyMethodDef ReCoDeMethods[] = 
{
	{ "create_buffers", (PyCFunction)create_buffers, METH_VARARGS, 0 },
	{ "get_frame_sparse", (PyCFunction)get_frame_sparse, METH_VARARGS, 0 },
	{ "bit_unpack_pixel_intensities", (PyCFunction)bit_unpack_pixel_intensities, METH_VARARGS, 0 },
	{ "bit_pack_pixel_intensities", (PyCFunction)bit_pack_pixel_intensities, METH_VARARGS, 0 },
	{0,0,0,0}
};

static struct PyModuleDef c_recode = {
	PyModuleDef_HEAD_INIT,
	"c_recode",   /* name of module */
	"Interface to C ReCoDe Reader", /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	ReCoDeMethods
};

static PyTypeObject ReCoDeReaderType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"c_recode.Reader",                            /*tp_name*/
	sizeof(RecodeReader),                          		/*tp_basicsize*/
	0,											/*tp_itemsize*/
	(destructor)ReCoDe_dealloc,						/*tp_dealloc*/
	0,                                          /*tp_print*/
	0,                                          /*tp_getattr*/
	0,                                          /*tp_setattr*/
	0,                                          /*tp_compare*/
	0,                                          /*tp_repr*/
	0,                                          /*tp_as_number*/
	0,                                          /*tp_as_sequence*/
	0,                                          /*tp_as_mapping*/
	0,                                          /*tp_hash */
	0,                                          /*tp_call*/
	0,                                          /*tp_str*/
	0,                                          /*tp_getattro*/
	0,                                          /*tp_setattro*/
	0,                                          /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,								/*tp_flags*/
	"C Recode Reader object",                               /*tp_doc*/
	0,											/*tp_traverse*/
	0,											/*tp_clear*/
	0,                                          /*tp_richcompare*/
	0,                                          /*tp_weaklistoffset*/
	0,                                          /*tp_iter*/
	0,                                          /*tp_iternext*/
	ReCoDeMethods,                                 /*tp_methods*/
	ReCoDe_members,                                /*tp_members*/
	0,                                          /*tp_getsets*/
	0,                                          /*tp_base*/
	0,                                          /*tp_dict*/
	0,                                          /*tp_descr_get*/
	0,                                          /*tp_descr_set*/
	0,                                          /*tp_dictoffset*/
	0,											/*tp_init*/
	0,                                          /*tp_alloc*/
	ReCoDe_new,										/*tp_new*/
};


PyMODINIT_FUNC
PyInit_c_recode(void)
{
	//return PyModule_Create(&c_recode);
	
	PyObject *m;
	if (PyType_Ready(&ReCoDeReaderType) < 0) {
		return NULL;
	}

	m = PyModule_Create(&c_recode);
	if (m == NULL) {
		return NULL;
	}

	Py_INCREF(&ReCoDeReaderType);
	if (PyModule_AddObject(m, "Reader", (PyObject *)&ReCoDeReaderType) < 0) {
        Py_DECREF(&ReCoDeReaderType);
        Py_DECREF(m);
        return NULL;
    }

	return m;
	
}