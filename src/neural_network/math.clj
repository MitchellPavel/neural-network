(ns neural-network.math
  (:require [clojure.math.numeric-tower :as math]))

(defn inverse-sqrt [x] (/ 1 (math/sqrt x)))

(defn sigmoid [x] (/ 1 ( + 1 (math/expt 2.7182818284590 (- x)))))

(defn sigmoid' [x] (* (sigmoid x) (- 1 (sigmoid x))))

(defn small-rand [x] (- (rand (* 2 x)) x))