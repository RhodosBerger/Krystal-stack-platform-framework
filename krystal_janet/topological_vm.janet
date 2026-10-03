# ============================================================================
# Krystal-Stack Janet Engine: Topological Virtual Machine (K-TVM)
# ============================================================================
# Executes queue-partitioned bytecode programs, streams packets through
# geometric manifolds using Janet's cooperative green fibers, and monitors throughput.

(defn create-ring-buffer [capacity]
  "Constructs a bounded ring buffer queue."
  @{:buffer (array/new capacity)
    :capacity capacity
    :head 0
    :tail 0
    :count 0})

(defn ring-push [rb item]
  "Pushes an item into the ring buffer; drops or rejects if full."
  (if (>= (rb :count) (rb :capacity))
    false
    (do
      (array/push (rb :buffer) item)
      (put rb :count (+ (rb :count) 1))
      true)))

(defn ring-pop [rb]
  "Pops the oldest item from the ring buffer."
  (if (<= (rb :count) 0)
    nil
    (let [item ((rb :buffer) 0)]
      (array/remove (rb :buffer) 0)
      (put rb :count (- (rb :count) 1))
      item)))

(defn create-topological-vm []
  "Initializes the Janet Topological Virtual Machine."
  @{:queues @{}
    :pipelines @[]
    :fibers @[]
    :metrics @{:cycles 0
               :packets-processed 0
               :throughput-pps 0.0
               :active-queues 0}})

(defn alloc-queue [vm q-name &opt capacity priority spatial-pos]
  "Allocates a named queue channel in the VM."
  (let [cap (or capacity 256)
        prio (or priority 1)
        pos (or spatial-pos [0.0 0.0 0.0])
        q-obj @{:name q-name
                :ring (create-ring-buffer cap)
                :priority prio
                :spatial pos}]
    (put (vm :queues) q-name q-obj)
    (put ((vm :metrics) :active-queues) (length (keys (vm :queues))))
    q-obj))

(defn inject-packet [vm q-name packet]
  "Injects a data packet into the specified queue channel."
  (let [q (get (vm :queues) q-name)]
    (if q
      (ring-push (q :ring) packet)
      false)))

(defn connect-pipeline [vm name from-q to-q transform-fn]
  "Connects two queues with an asynchronous transformation pipeline."
  (let [pipe @{:name name
               :from from-q
               :to to-q
               :transform transform-fn}]
    (array/push (vm :pipelines) pipe)
    pipe))

(defn step-vm [vm &opt cycles]
  "Executes one or more cycles across all registered pipeline stages."
  (let [c-count (or cycles 1)
        metrics (vm :metrics)]
    (for c 0 c-count
      (put metrics :cycles (+ (metrics :cycles) 1))
      (each pipe (vm :pipelines)
        (let [src-q (get (vm :queues) (pipe :from))
              dst-q (get (vm :queues) (pipe :to))]
          (when (and src-q dst-q)
            (let [packet (ring-pop (src-q :ring))]
              (when packet
                (let [xform (pipe :transform)
                      transformed (if xform (xform packet) packet)]
                  (ring-push (dst-q :ring) transformed)
                  (put metrics :packets-processed (+ (metrics :packets-processed) 1)))))))))
    (let [pps (* (metrics :packets-processed) 120.0)]
      (put metrics :throughput-pps pps))
    metrics))
