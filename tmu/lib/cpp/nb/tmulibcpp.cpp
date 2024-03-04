//
// Created by per on 3/1/24.
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/map.h>

#include <iostream>

#include "tm_memory.h"
#include "tm_clause_dense.h"
#include "tm_weight_bank.h"
#include "models/classifiers/tm_vanilla.h"
#include "utils/sparse_clause_container.h"
#include <tl/optional.hpp>


extern "C" {
    #include "ClauseBank.h"
    #include"Tools.h"
}

using namespace nanobind;
using namespace nanobind::literals;
namespace nb = nanobind;


NAMESPACE_BEGIN(NB_NAMESPACE)
    NAMESPACE_BEGIN(detail)

        template <typename T> struct remove_opt_mono<tl::optional<T>>
                : remove_opt_mono<T> { };

        template <typename T>
        struct type_caster<tl::optional<T>> {
            using Caster = make_caster<T>;

            NB_TYPE_CASTER(tl::optional<T>, const_name("Optional[") +
                                            concat(Caster::Name) +
                                            const_name("]"))

            type_caster() : value(tl::nullopt) { }

            bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
                if (src.is_none()) {
                    value = tl::nullopt;
                    return true;
                }

                Caster caster;
                if (!caster.from_python(src, flags, cleanup))
                    return false;

                static_assert(
                        !std::is_pointer_v<T> || is_base_caster_v<Caster>,
                        "Binding ``optional<T*>`` requires that ``T`` is handled "
                        "by nanobind's regular class binding mechanism. However, a "
                        "type caster was registered to intercept this particular "
                        "type, which is not allowed.");

                value.emplace(caster.operator cast_t<T>());

                return true;
            }

            template <typename T_>
            static handle from_cpp(T_ &&value, rv_policy policy, cleanup_list *cleanup) noexcept {
                if (!value)
                    return none().release();

                return Caster::from_cpp(forward_like<T_>(*value), policy, cleanup);
            }
        };

        template <> struct type_caster<tl::nullopt_t> {
            bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
                if (src.is_none())
                    return true;
                return false;
            }

            static handle from_cpp(tl::nullopt_t, rv_policy, cleanup_list *) noexcept {
                return none().release();
            }

            NB_TYPE_CASTER(tl::nullopt_t, const_name("None"))
        };

    NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)


template <typename T>
nb::class_<SparseClauseContainer<T>>& bind_sparse_container(const char *name, nb::module_& m){
    return nb::class_<SparseClauseContainer<T>>(m, name)
            .def("__getitem__", &SparseClauseContainer<T>::operator[])
            .def("__len__", &SparseClauseContainer<T>::size)
            .def("classes", &SparseClauseContainer<T>::get_classes)
            .def("sample", &SparseClauseContainer<T>::sample,  "exclude"_a = std::set<int>())
            .def("set_clause_init", [](
                    SparseClauseContainer<T>& self,
                    const nb::object& cls,
                    const nb::dict& args
            ) {
                auto instance = cls(**args); // Call the Python callable to create an instance
                auto ins = nb::cast<std::shared_ptr<T>>(instance);
                self.template_instance = ins;

            })
            .def("populate", &SparseClauseContainer<T>::populate);
}



NB_MODULE(tmulibpy, m) {

    bind_sparse_container<TMClauseBankDense<uint32_t>>("SparseClauseContainer", m)
            .def(nb::init<unsigned int>(), "random_seed"_a)

    ;
    bind_sparse_container<TMWeightBank<uint32_t>>("SparseWeightContainer", m)
            .def(nb::init<unsigned int>(), "random_seed"_a)
    ;

    nanobind::class_<TMMemory<uint32_t>>(m, "TMMemory")
        .def(nanobind::init<>())
        .def("reserve", &TMMemory<uint32_t>::reserve)
        .def("push_back", &TMMemory<uint32_t>::push_back)
    ;

    nb::class_<TMVanillaClassifier<uint32_t>>(m, "TMVanillaClassifier")
        .def(nb::init<
            int,
            float,
            float,
            uint32_t,
            bool,
            bool,
            bool,
            bool,
            bool,
            float,
            uint32_t,
            bool,
            bool,
            tl::optional<std::vector<int>>,
            int32_t ,
            int32_t,
            int32_t,
            bool,
            int
        >(),
            "T"_a,
            "s"_a,
            "d"_a,
            "number_of_clauses"_a,
            "confidence_driven_updating"_a,
            "weighted_clauses"_a,
            "type_i_feedback"_a,
            "type_ii_feedback"_a,
            "type_iii_feedback"_a,
            "type_i_ii_ratio"_a,
            "max_included_literals"_a,
            "boost_true_positive_feedback"_a,
            "reuse_random_feedback"_a,
            "patch_dim"_a = std::nullopt,
            "number_of_state_bits"_a,
            "number_of_state_bits_ind"_a,
            "batch_size"_a,
            "incremental"_a,
            "seed"_a
        )
        .def_ro("memory", &TMVanillaClassifier<uint32_t>::memory)
        .def("get_required_memory_size", &TMVanillaClassifier<uint32_t>::get_required_memory_size)
        .def("init_after", [](TMVanillaClassifier<uint32_t>& self, const nb::ndarray<>& X, const nb::ndarray<>& Y){
            std::cout << "init_after (TODO handling numpy arrays)" << std::endl;
            self.init_after();
        })
        .def("initialize", &TMVanillaClassifier<uint32_t>::initialize)
        .def("init", [](TMVanillaClassifier<uint32_t>& self, nb::ndarray<uint32_t>& X, nb::ndarray<uint32_t>& Y){

            std::vector<int> X_shape = {static_cast<int>(X.shape(0)), static_cast<int>(X.shape(1))};
            auto x = tcb::span(X.data(), X.size());
            auto y = tcb::span(Y.data(), Y.size());

            self.init(y, x, X_shape);
        })
        .def(
            "mechanism_compute_update_probabilities",
            &TMVanillaClassifier<uint32_t>::mechanism_compute_update_probabilities,
            "is_target"_a,
            "class_sum"_a
        )
        .def(
        "mechanism_feedback",
        [](
                TMVanillaClassifier<uint32_t>& self,
                bool is_target,
                uint32_t target,
                nanobind::ndarray<uint32_t , nb::ndim<1>, c_contig>& clause_outputs,
                float update_p,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& clause_active,
                nanobind::ndarray<uint32_t, nb::ndim<1>, c_contig>& literal_active,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X_train,
                int sample_idx
        ){

            uint32_t* encoded_xi = &encoded_X_train(sample_idx, 0);
            auto encoded_xi_len = encoded_X_train.shape(1);
            auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);
            auto span_clause_outputs = tcb::span(clause_outputs.data(), clause_outputs.size());
            auto span_clause_active = tcb::span(clause_active.data(), clause_active.size());
            auto span_literal_active = tcb::span(literal_active.data(), literal_active.size());

            self.mechanism_feedback(
                    is_target,
                    target,
                    span_clause_outputs,
                    update_p,
                    span_clause_active,
                    span_literal_active,
                    span_encoded_xi
            );

        },
        "is_target"_a,
        "target"_a,
        "clause_outputs"_a,
        "update_p"_a,
        "clause_active"_a,
        "literal_active"_a,
        "encoded_X_train"_a,
        "sample_idx"_a
        )
        .def("predict", [](
                TMVanillaClassifier<uint32_t>& self,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X_test,
                bool clip_class_sum = false,
                bool return_class_sum = false) {

            auto encoded_X_test_span = tcb::span(encoded_X_test.data(), encoded_X_test.size());
            std::vector<int> X_shape = {static_cast<int>(encoded_X_test.shape(0)), static_cast<int>(encoded_X_test.shape(1))};

            auto [argmax, class_sums] = self.predict(
                    encoded_X_test_span,
                    X_shape,
                    clip_class_sum,
                    return_class_sum
            );


            return std::make_tuple(argmax, class_sums);


        }, nb::rv_policy::take_ownership)



            .def("fit",
        [](
                TMVanillaClassifier<uint32_t>& self,
                nanobind::ndarray<uint32_t, nb::ndim<1>, c_contig>& y,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X_train
        ) {

            auto y_span = tcb::span(y.data(), y.size());
            auto encoded_X_train_span = tcb::span(encoded_X_train.data(), encoded_X_train.size());

            std::vector<int> X_shape = {static_cast<int>(encoded_X_train.shape(0)), static_cast<int>(encoded_X_train.shape(1))};
            self.fit(
                    y_span,
                    encoded_X_train_span,
                    X_shape,
                    true
            );


        },
        "y"_a,
        "encoded_X_train"_a,
         nb::call_guard<nb::gil_scoped_release>()
        )

        .def("predict_compute_class_sums", [](
                TMVanillaClassifier<uint32_t>& self,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X_test,
                int ith_sample,
                bool clip_class_sum
        ) {

            auto encoded_X_test_span = tcb::span(encoded_X_test.data(), encoded_X_test.size());
            auto num_items = encoded_X_test.shape(0);

            return self.predict_compute_class_sums(
                    encoded_X_test_span,
                    ith_sample,
                    num_items,
                    clip_class_sum
            );


        }, "encoded_X_test"_a, "ith_sample"_a, "clip_class_sum"_a)

        .def_prop_ro("positive_clauses", [](TMVanillaClassifier<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.positive_clauses.data(),
                    {static_cast<unsigned long>(self.number_of_clauses)}
            );
        }, nb::rv_policy::reference)
        .def_prop_ro("negative_clauses", [](TMVanillaClassifier<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.negative_clauses.data(),
                    {static_cast<unsigned long>(self.number_of_clauses)}
            );
        }, nb::rv_policy::reference)
        .def_prop_ro("clause_banks", [](TMVanillaClassifier<uint32_t>& self) {
            return &self.clause_banks;
        }, nb::rv_policy::reference)
        .def_prop_ro("weight_banks", [](TMVanillaClassifier<uint32_t>& self) {
            return &self.weight_banks;
        }, nb::rv_policy::reference)

        ;


    nb::class_<TMWeightBank<uint32_t>>(m, "TMWeightBank")
            .def(nb::init<>())
            .def("initialize", &TMWeightBank<uint32_t>::initialize)
            .def("get_weights", [](TMWeightBank<uint32_t>& self) {
                return nb::ndarray<nb::numpy, uint32_t>(
                        self.weights.data(),
                        {static_cast<unsigned long>(self.weights.size())}
                );
            }, nb::rv_policy::reference)
            .def("get_required_memory_size", &TMWeightBank<uint32_t>::getRequiredMemorySize)
            .def("increment", [](
                    TMWeightBank<uint32_t>& self,
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, c_contig>& clause_output,
                    float update_p,
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, c_contig>& clause_active,
                    bool positive_weights
            ) {
                tcb::span<uint32_t> clause_output_span(clause_output.data(), clause_output.size());
                tcb::span<uint32_t> clause_active_span(clause_active.data(), clause_active.size());
                self.increment(clause_output_span, update_p, clause_active_span, positive_weights);
            },
             "clause_output"_a,
             "update_p"_a,
             "clause_active"_a,
             "positive_weights"_a
            )
            .def("decrement", [](
                    TMWeightBank<uint32_t>& self,
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, c_contig>& clause_output,
                    float update_p,
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, c_contig>& clause_active,
                    bool negative_weights
            ) {
                tcb::span<uint32_t> clause_output_span(clause_output.data(), clause_output.size());
                tcb::span<uint32_t> clause_active_span(clause_active.data(), clause_active.size());
                self.decrement(clause_output_span, update_p, clause_active_span, negative_weights);
            },
             "clause_output"_a,
             "update_p"_a,
             "clause_active"_a,
             "negative_weights"_a)

            ;


    nanobind::class_<TMClauseBankDense<uint32_t>>(m, "TMClauseBankDense")
        .def(nanobind::init<
                     float,
                     float,
                     bool,
                     bool,
                     std::vector<int>,
                     tl::optional<std::vector<int>>,
                     tl::optional<std::size_t>,
                     std::size_t,
                     std::size_t,
                     std::size_t,
                     std::size_t,
                     bool,
                     int
             >(),
             "s"_a,
             "d"_a,
             "boost_true_positive_feedback"_a,
             "reuse_random_feedback"_a,
             "X_shape"_a,
             "patch_dim"_a = std::nullopt,
             "max_included_literals"_a = std::nullopt,
             "number_of_clauses"_a,
             "number_of_state_bits"_a,
             "number_of_state_bits_ind"_a,
             "batch_size"_a,
             "incremental"_a,
             "seed"_a = 0
        )
        .def("initialize", &TMClauseBankDense<uint32_t>::initialize)
        .def("set_ta_state", &TMClauseBankDense<uint32_t>::setTAState)
        .def("get_ta_state", &TMClauseBankDense<uint32_t>::getTAState)
        .def_ro("clause_output", &TMClauseBankDense<uint32_t>::clause_output)
        .def_ro("clause_output_batch", &TMClauseBankDense<uint32_t>::clause_output_batch)
        .def_ro("clause_and_target", &TMClauseBankDense<uint32_t>::clause_and_target)
        .def_ro("clause_output_patchwise", &TMClauseBankDense<uint32_t>::clause_output_patchwise)
        .def_ro("feedback_to_ta", &TMClauseBankDense<uint32_t>::feedback_to_ta)
        .def_ro("output_one_patches", &TMClauseBankDense<uint32_t>::output_one_patches)
        .def_ro("literal_clause_count", &TMClauseBankDense<uint32_t>::literal_clause_count)
        .def_ro("type_ia_feedback_counter", &TMClauseBankDense<uint32_t>::type_ia_feedback_counter)
        .def_ro("literal_clause_map", &TMClauseBankDense<uint32_t>::literal_clause_map)
        .def_ro("literal_clause_map_pos", &TMClauseBankDense<uint32_t>::literal_clause_map_pos)
        .def_ro("false_literals_per_clause", &TMClauseBankDense<uint32_t>::false_literals_per_clause)
        .def_ro("previous_xi", &TMClauseBankDense<uint32_t>::previous_xi)
        .def_ro("clause_bank", &TMClauseBankDense<uint32_t>::clause_bank)
        .def_ro("actions", &TMClauseBankDense<uint32_t>::actions)
        .def_ro("clause_bank_ind", &TMClauseBankDense<uint32_t>::clause_bank_ind)
        .def_ro("number_of_ta_chunks", &TMClauseBankDense<uint32_t>::number_of_ta_chunks)
        .def_ro("number_of_patches", &TMClauseBankDense<uint32_t>::number_of_patches)
        .def_ro("dim", &TMClauseBankDense<uint32_t>::dim)
        .def_ro("patch_dim", &TMClauseBankDense<uint32_t>::patch_dim)
        .def_ro("number_of_clauses", &TMClauseBankDense<uint32_t>::number_of_clauses)
        .def_ro("number_of_state_bits", &TMClauseBankDense<uint32_t>::number_of_state_bits)
        .def_ro("number_of_state_bits_ind", &TMClauseBankDense<uint32_t>::number_of_state_bits_ind)
        .def_ro("batch_size", &TMClauseBankDense<uint32_t>::batch_size)
        .def_ro("number_of_literals", &TMClauseBankDense<uint32_t>::number_of_literals)


        .def("get_clause_bank", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t, nb::c_contig>(
                    self.clause_bank.data(),
                    {static_cast<unsigned long>(self.calculateClauseBankSize())}
            );
        }, nb::rv_policy::reference)

        .def("get_clause_bank_ind", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.clause_bank_ind.data(),
                    {static_cast<unsigned long>(self.calculateClauseBankSize())}
            );
        }, nb::rv_policy::reference)

        .def("get_clause_output", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.clause_output.data(),
                    {static_cast<unsigned long>(self.number_of_clauses)}
            );
        }, nb::rv_policy::reference)

        .def("get_clause_output_batch", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.clause_output_batch.data(),
                    {static_cast<unsigned long>(self.calculateClauseOutputBatchSize())}
            );
        }, nb::rv_policy::reference)

        .def("get_clause_output_patchwise", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.clause_output_patchwise.data(),
                    {static_cast<unsigned long>(self.number_of_clauses * self.number_of_patches)}
            );
        }, nb::rv_policy::reference)

        .def("get_feedback_to_ta", [](TMClauseBankDense<uint32_t>& self) {
            return nb::ndarray<nb::numpy, uint32_t>(
                    self.feedback_to_ta.data(),
                    {static_cast<unsigned long>(self.number_of_clauses)}
            );
        }, nb::rv_policy::reference)

        .def("get_required_memory_size", &TMClauseBankDense<uint32_t>::getRequiredMemorySize)
        .def("prepare_X", [](
                     TMClauseBankDense<uint32_t>& self,
                     nb::ndarray<uint32_t>& x) {

                 if (x.ndim() != 2) { // Assuming x should be a 2D array, adjust as needed
                     throw std::runtime_error("Input array x must be a 2D array");
                 }

                 auto number_of_examples = x.shape(0);

                 auto* encoded_X = new uint32_t[number_of_examples * self.number_of_patches * self.number_of_ta_chunks];
                 nb::capsule owner(encoded_X, [](void* p) noexcept {
                     delete[] static_cast<uint32_t*>(p);
                 });


                 tmu_encode(
                         x.data(),
                         encoded_X,
                         static_cast<int>(number_of_examples),
                         std::get<0>(self.dim),
                         std::get<1>(self.dim),
                         std::get<2>(self.dim),
                         std::get<0>(self.patch_dim),
                         std::get<1>(self.patch_dim),
                         1,
                         0
                 );

                 return nb::ndarray<nb::numpy, uint32_t, c_contig>(
                         encoded_X,
                         {number_of_examples, self.number_of_patches * self.number_of_ta_chunks},
                         owner
                 );
             },
             "x"_a, nb::rv_policy::take_ownership)


         .def("calculate_clause_outputs_update", [](
                TMClauseBankDense<uint32_t>& self,
                nanobind::ndarray<uint32_t, nb::ndim<1>, c_contig>& literal_active,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X,
                int e){

            auto encoded_xi = &encoded_X(e, 0);
            auto encoded_xi_len = encoded_X.shape(1);
            auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);

            auto literal_active_span = tcb::span(literal_active.data(), literal_active.size());

            self.calculate_clause_outputs_update(
                    literal_active_span,
                    span_encoded_xi
            );


            return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig>(
                  self.clause_output.data(),
                  {static_cast<unsigned long>(self.number_of_clauses)
            });

         }
         , "literal_active"_a
         , "encoded_X"_a
         , "e"_a
         , nb::rv_policy::automatic_reference)

         .def("type_i_feedback", [](
                 TMClauseBankDense<uint32_t>& self,
                 float update_p,
                 nb::ndarray<uint32_t>& clause_active,
                 nb::ndarray<uint32_t>& literal_active,
                 nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X,
                 int e
                 ) {

            auto clause_active_span = tcb::span(clause_active.data(), clause_active.size());
            auto literal_active_span = tcb::span(literal_active.data(), literal_active.size());
            auto encoded_xi = &encoded_X(e, 0);
            auto encoded_xi_len = encoded_X.shape(1);
            auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);

            self.type_i_feedback(
                    update_p,
                    clause_active_span,
                    literal_active_span,
                    span_encoded_xi
            );
         }
            , "update_p"_a
            , "clause_active"_a
            , "literal_active"_a
            , "encoded_X"_a
            , "e"_a
         , nb::rv_policy::automatic
         )

    .def("type_ii_feedback", [](
                 TMClauseBankDense<uint32_t>& self,
                 float update_p,
                 nb::ndarray<uint32_t>& clause_active,
                 nb::ndarray<uint32_t>& literal_active,
                 nanobind::ndarray<uint32_t, nb::ndim<2>, nb::c_contig>& encoded_X,
                 uint32_t target) {

                auto clause_active_span = tcb::span(clause_active.data(), clause_active.size());
                auto literal_active_span = tcb::span(literal_active.data(), literal_active.size());
                auto encoded_xi = &encoded_X(target, 0);
                auto encoded_xi_len = encoded_X.shape(1);
                auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);

                 self.type_ii_feedback(
                    update_p,
                    clause_active_span,
                    literal_active_span,
                    span_encoded_xi
                 );
         }

            , "update_p"_a
            , "clause_active"_a
            , "literal_active"_a
            , "encoded_X"_a
            , "e"_a
            , nb::rv_policy::automatic
    )

    .def("calculate_clause_outputs_update", [](
            TMClauseBankDense<uint32_t>& self,
                nanobind::ndarray<uint32_t>& literal_active,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X,
                uint32_t e) {

        auto encoded_xi = &encoded_X(e, 0);
        auto encoded_xi_len = encoded_X.shape(1);
        auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);
        auto literal_active_span = tcb::span(literal_active.data(), literal_active.size());

        self.calculate_clause_outputs_update(
                literal_active_span,
                span_encoded_xi
        );

         return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig>(
                 self.clause_output.data(),
                 {self.number_of_clauses}
         );
    }
    , "literal_active"_a
    , "encoded_X"_a
    , "e"_a
    , nb::rv_policy::automatic_reference)


    .def("calculate_clause_outputs_predict", [](
                TMClauseBankDense<uint32_t>& self,
                nanobind::ndarray<uint32_t, nb::ndim<2>, c_contig>& encoded_X,
                int sample_index) {

        auto encoded_xi = &encoded_X(sample_index, 0);
        auto encoded_xi_len = encoded_X.shape(1);
        auto n_items = encoded_X.shape(0);
        auto span_encoded_xi = tcb::span(encoded_xi, encoded_xi_len);

        auto clause_outputs = self.calculate_clause_outputs_predict(
                span_encoded_xi,
                sample_index,
                n_items
        );

         auto arr = nb::ndarray<nb::numpy, uint32_t , nb::ndim<1>, nb::c_contig>(
                 clause_outputs.data(),
                 {self.number_of_clauses}
         );

        return arr;

    }
    , "encoded_X"_a
    , "e"_a
    , nb::rv_policy::reference)


    ;




}