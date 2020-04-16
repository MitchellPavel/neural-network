(ns neural-network.io)

(defn byte->unsigned-byte
  [b]
  (bit-and b 0x0FF))

(defn bytes->num
  [data]
  (reduce
    bit-or
    (map-indexed
      (fn [i x]
        (bit-shift-left (byte->unsigned-byte x) (* 8 (- (count data) i 1))))
      data)))

(defn get-processor
  "gets the data type"
  [data-type]
  (case data-type
    0x08; unsigned byte
      (fn [binary-data] (map byte->unsigned-byte binary-data))
    0x09; signed byte
      (fn [binary-data] binary-data)
    0x0B; short (2 bytes)
      (fn [binary-data] (map bytes->num (partition 2 binary-data)))
    0x0C; int (4 bytes)
      (fn [binary-data] (map bytes->num (partition 4 binary-data)))
    0x0D; float
      (throw (Exception. "I do not yet know how to convet binary to float" ))
    0x0E; double
      (throw (Exception. "I do not yet know how to convet binary to double"))
    (throw (Exception. (format "Could not recognize data type %x" data-type)))))

(defn get-dims
  "get the size of all the dimensions"
  [binary-data]
  (map
    bytes->num
    (partition 4 (subvec binary-data 2 (+ 2 (* 4 (second binary-data)))))))

(defn partition-on-dimensions
  "partition on the dims"
  [dimensions data]
  (cond
    (empty? dimensions)
      data
    (empty? (rest dimensions))
      (partition
        (first dimensions)
        data)
    :else
      (partition
        (first dimensions)
        (partition-on-dimensions (rest dimensions) data))))

(defn get-data
  "return all the data"
  [path]
  (def binary-data (with-open
    [in (java.util.zip.GZIPInputStream.
          (clojure.java.io/input-stream path))
    out (java.io.ByteArrayOutputStream.)]
    (clojure.java.io/copy in out)
    (vec (drop 2 (.toByteArray out)))))
  (partition-on-dimensions
    (rest (get-dims binary-data))
    ((get-processor (first binary-data))
      (subvec binary-data (+ 2 (* 4 (second binary-data)))))))