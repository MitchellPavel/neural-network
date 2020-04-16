(defproject neural-network "0.1.0-SNAPSHOT"
  :description "Simple neural networks for recognizing handwriting"
  :url "https://github.com/MitchellPavel/neural-network"
  :license {:name "MIT"
            :url "https://raw.githubusercontent.com/MitchellPavel/neural-network/master/LICENSE"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [net.mikera/core.matrix "0.62.0"]]
  :repl-options {:init-ns neural-network.core}
  :main main)