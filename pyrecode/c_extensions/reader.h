
#define SetBit(A,k)		( A[(k/8)] |= (1 << (k%8)) )
#define ClearBit(A,k)	( A[(k/8)] &= ~(1 << (k%8)) )
#define CheckBit(A,k)	( A[(k/8)] & (1 << (k%8)) )

/*
Called by the python function get_frame_sparse
this function is called by pyrecode_c.py
*/
int64_t _unpack_frame_sparse(
	uint16_t nx, uint16_t ny, uint8_t bit_depth,
	uint8_t *deCompressedBinaryImage, uint8_t *deCompressedPixvals,
	uint64_t *frameBuffer,
	uint8_t reduction_level
) {

	uint16_t row, col;
	uint32_t linear_pixel_index, pixel_bit_index_frame;
	uint64_t extracted_pixval;
	uint8_t n;
	uint64_t n_fg_pixels = 0;

	for (row = 0; row < ny; row++) {
		for (col = 0; col < nx; col++) {
			linear_pixel_index = row * nx + col;
			if (CheckBit(deCompressedBinaryImage, linear_pixel_index) > 0) {

			    if (reduction_level == 1) {
                    // unpack pixval
                    extracted_pixval = 0;
                    // Assumes LITTLE-ENDIAN Byte Order of pixval
                    for (n = 0; n < bit_depth; n++) {
                        pixel_bit_index_frame = n_fg_pixels*bit_depth + n;
                        if (CheckBit(deCompressedPixvals, pixel_bit_index_frame) > 0) {
                            //extracted_pixval += pow2_lookup_table[n];
                            extracted_pixval |= 1ULL << n;
                        }
                    }
				} else {
				    extracted_pixval = 1;
				}

                /*==============DEBUG ONLY================
				if (row < 5 && col < 5) {
				    printf("%d,%d: ", row, col);
				    for (n = 0; n < bit_depth; n++) {
                        pixel_bit_index_frame = n_fg_pixels*bit_depth + n;
                        if (CheckBit(deCompressedPixvals, pixel_bit_index_frame) > 0) {
                            printf("1 ");
                        } else {
                            printf("0 ");
                        }
                    }
                    printf(" = %lu\n", extracted_pixval);
				}
				*==============DEBUG ONLY===============*/

				frameBuffer[n_fg_pixels * 3] = row;
				frameBuffer[n_fg_pixels * 3 + 1] = col;
				frameBuffer[n_fg_pixels * 3 + 2] = extracted_pixval;

				n_fg_pixels++;
			}
		}
	}
	//printf("Decoded Frame with %d foreground pixels\n", n_fg_pixels);
	return (int64_t)n_fg_pixels;
}

/*
Called by the python function _bit_pack_pixel_intensities
a similar function with scaling is declared above: scale_and_pack_pixvals
*/
float _bit_unpack_pixel_intensities (
	uint64_t n_values,
	uint8_t bit_depth,
	uint8_t *deCompressedPixvals,
	uint64_t *buffer
) {

    uint8_t n;
    uint64_t v;
    uint64_t pixel_bit_index_frame;
    uint64_t extracted_pixval[1];

    for (v = 0; v < n_values; n++) {
        extracted_pixval[0] = 0;
        for (n = 0; n < bit_depth; n++) {
            pixel_bit_index_frame = v*bit_depth + n;
            if (CheckBit(deCompressedPixvals, pixel_bit_index_frame) > 0) {
                //extracted_pixval += pow2_lookup_table[n];
                SetBit(extracted_pixval, n);
            }
        }
        buffer[v] = extracted_pixval[0];
    }

    return 0;
}

/*
Called by the python function _bit_pack_pixel_intensities
a similar function with scaling is declared above: scale_and_pack_pixvals
*/
float _bit_pack_pixel_intensities (
						uint64_t sz_packedPixval, 
						uint32_t n_fg_pixels,
						uint8_t  bit_depth, 
						uint16_t *pixvals, 
						uint8_t  *packedPixvals) {
	
	clock_t p_start = clock();

    uint64_t p;
	int n, linear_index, nth_bit_of_pth_pixval;
	
	// setting packedPixvals to 0 in a for loop is faster than using ClearBit
	for (p = 0; p < sz_packedPixval; p++) {
		packedPixvals[p] = 0;
	}
	
	for (p=0; p<n_fg_pixels; p++) {
		
		// Assumes LITTLE-ENDIAN Byte Order
		for (n=0; n<bit_depth; n++) {
			nth_bit_of_pth_pixval = (pixvals[p] & ( 1 << n )) >> n;
			if (nth_bit_of_pth_pixval != 0) {
				linear_index = p*bit_depth + n;
				SetBit(packedPixvals, linear_index);
			}
		}
		
		//printf("%f, %hu, %hu, %hu, %hu\n", pixval_01, scaled_pixval, pixvals[p], data_min, data_max);
	}
	
	clock_t p_end = clock();
	float process_time = (p_end - p_start) * 1000.0 / CLOCKS_PER_SEC;
	return process_time;
	
}
