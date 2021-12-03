# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
    void cb_calculate_clause_outputs_predict(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi);
    void cb_calculate_clause_outputs_update(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi);
    void cb_calculate_clause_outputs_patchwise(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi);
    void cb_type_i_feedback(unsigned int *ta_state, unsigned int *feedback_to_ta, unsigned int *output_one_patches, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, float update_p, float s, unsigned int boost_true_positive_feedback, unsigned int *clause_active, unsigned int *Xi);
    void cb_type_ii_feedback(unsigned int *ta_state, unsigned int *output_one_patches, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, float update_p, unsigned int *clause_active, unsigned int *Xi);
    """)

ffibuilder.set_source("tmu._cb",  # name of the output C extension
"""
    #include "./tmu/ClauseBank.h"
""",
    include_dirs=['.'],
    sources=['./tmu/ClauseBank.c'],   # includes pi.c as additional sources
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
