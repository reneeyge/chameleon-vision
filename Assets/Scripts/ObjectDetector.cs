using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class ObjectDetector : MonoBehaviour
{
    public NNModel modelAsset;
    private Model m_RuntimeModel;

    // Start is called before the first frame update
    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        //m_Worker = WorkerFactory.CreateWorker(<WorkerFactory.Type>, m_RuntimeModel);
    }

    // Update is called once per frame
    void Update()
    {
        Tensor input = new Tensor(batch, height, width, channels);
        m_Worker.Execute(input);
        Tensor 0 = m_Worker.PeekOutput("output_layer_name");
        input.Dispose();
    }
}

/*
pseudocode

- internal objects (resources, buffer, workers)
- public constructor (like the models and bounding boxes??)
- dispose implementation
- public accessors ??
- main image processing function
- gpu to cpu readback function

*/