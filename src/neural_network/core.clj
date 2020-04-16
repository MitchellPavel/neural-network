(ns neural-network.core
  (:require
   [neural-network.math :refer [inverse-sqrt sigmoid sigmoid' small-rand]]
   [clojure.core.matrix :as m]))

(m/set-current-implementation :ndarray)

(defn create-nn
  [[x n :as spec]]
  (if-not n '()
    (let [d (inverse-sqrt x)
          w (m/matrix (repeatedly n #(repeatedly x (partial small-rand d))))
          b (m/zero-vector n)]
      (cons {:w w :b b} (create-nn (rest spec))))))

(defn create-delta
  [{w :w}]
  (let [r (m/row-count w) c (m/column-count w)]
    {:delta-w (m/zero-matrix r c) :delta-b (m/zero-vector r)}))

(defn eval-nn
  [in [{:keys [w b] :as l} :as nn]]
  (if (empty? nn) '()
    (let [z (m/add b (m/mmul w in))
          a (m/emap sigmoid z)]
      (cons (assoc l :z z :a a) (eval-nn a (rest nn))))))

(defn back-propogate
  [[{w :w} {a :a z :z} :as result] delta]
    (let [l {:delta-b delta :delta-w (m/outer-product delta a)}]
      (if-not z (list l)
        (cons l
          (back-propogate
            (rest result)
            (m/emul (m/mmul (m/transpose w) delta) (m/emap sigmoid' z)))))))

(defn eval-delta
  [[data label] nn]
  (let [[{:keys [a z]} :as result] (reverse (cons {:a data} (eval-nn data nn)))
        delta (m/mul 2 (m/sub a label) (m/emap sigmoid' z))]
    (reverse (back-propogate result delta))))

(defn apply-delta
  [e {:keys [delta-b delta-w]} {:keys [w b]}]
  {:b (m/sub b (m/mul e delta-b))
   :w (m/sub w (m/mul e delta-w))})

(defn sum-delta
  [{dw1 :delta-w db1 :delta-b} {dw2 :delta-w db2 :delta-b}]
  {:delta-w (m/add dw1 dw2) :delta-b (m/add db1 db2)})

(defn pos-of-max [v] (first (apply max-key second (map-indexed vector v))))

(defn success?
  [nn [data label]]
  (if (= (pos-of-max label) (pos-of-max (:a (last (eval-nn data nn))))) 1 0))

(defn test-nn
  [nn test-data]
  (/ (reduce + (map #(success? nn %) test-data)) (count test-data)))

;; TODO: take in a config map
;; TODO: include option for a learning rate in the config
;; TODO: include option for a learning rate function in the config
;; TODO: include option for batch size in the config and take all the images
;; TODO: include option for the cost function in the config
;; TODO: include option for the non-linear function in the config
(defn train
  [nn training-data]
  (println (test-nn nn training-data))
  (map
    #(apply-delta (/ 0.5 (count training-data)) %1 %2)
    (reduce
      #(map (partial sum-delta) %1 (eval-delta %2 nn))
      (map create-delta nn)
      training-data)
    nn))

