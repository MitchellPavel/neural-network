(ns main
  (:gen-class)
  (:require
   [neural-network.core :as nn]
   [neural-network.io :as io]
   [clojure.core.matrix :as m]))

;; TODO: reorganize namespaces
;; TODO: write tests
;; TODO: reorganize API (including the input file)
;; TODO: review anonymous function syntax and add syntactic sugar as desired
;; TODO: break functions down so that they are more "pure"
;; TODO: put this out on GitHub
;; TODO: add a pipeline
;; TODO: add pre-conditions where applicable
;; TODO: get a real logger
;; TODO: add config with plain EDN file or outpace/config
;; TODO: make it more performant with concurrency maybe?
;; TODO: make train more flexible with config options
;; TODO: open source the thing and write an article
;; TODO: make CLI
;; TODO: add a graphql or REST API
;; TODO: deploy it via AWS
;; TODO: tie it into a django/python app with some pages
;; TODO: add an Angular UI
;; TODO: publish it as a helm-chart

(defn num->vec [num] (vec (concat (repeat num 0) [1] (repeat (- 9 num) 0))))

(defn -main [training-images training-labels]
  (def training-data (partition 2 (interleave
    (map m/matrix (map flatten (io/get-data training-images)))
    (map m/matrix (map num->vec (io/get-data training-labels))))))
  (def n
    (reduce
      nn/train
      (nn/create-nn (list (* 28 28) 500 150 10))
      (partition 100 training-data))))

; input file contains:
;   neural-network:
;     - layer-spec
;   train:
;     - training-options
;     - training-images
;     - training-labels
;   test:
;     - test-images
;     - test-labels
;     - output-options

;; namespaces I see here are core.test-nn, core.train-nn, core.create-nn and io.load io.out
; (defn -main [input-file]
;   (let [{opts...} (load-options input-file)]
;     (output-nn :opts some-output-opts
;       (test-nn :opts some-testing-opts
;         (load-data some-testing-opts)
;         (train-nn :opts some-training-opts
;           (load-data some-training-opts)
;           (create-nn :opts some-basic-opts))))))
