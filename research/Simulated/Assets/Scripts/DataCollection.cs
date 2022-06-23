using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Threading;

public class DataCollection : MonoBehaviour
{
    static float MAXTIME = 180;
    static int FPS = 60;

    [SerializeField] ParameterData data;
    [SerializeField] ActionController actionController;
    [SerializeField] Savepath savepath;
    [SerializeField] bool recording;

    List<Frame> frames;
    int frame;
    float time;
    int maxFrames;
    bool busy;

    void Awake() {
        Time.captureFramerate = FPS;
        maxFrames = (int)MAXTIME * FPS;
    }

    void Start() {
        Initialize();
    }

    void OnDisable() {
        if(busy)
            savepath.Cancel(false);
    }

    void Initialize() {
        actionController.Initialize(data);
        frames = new List<Frame>();
        time = 0;
        frame = 0;
        if(recording) {
            savepath.Initialize();
            WriteMeta();
            busy = true;
        }
    }

    void Update() {
        if(Input.GetKeyDown(KeyCode.R))
            Initialize();

        time += Time.deltaTime;
        frame++;
        if(recording)
            WriteFrame();
        if(frame > maxFrames) {
            if(recording)
                FinalizeFrames();
            Initialize();
        }
    }

    void WriteMeta() {
        File.WriteAllText(savepath.metapath, JsonUtility.ToJson(data));
    }

    void WriteFrame() {
        Frame frame = new Frame() { frame = this.frame, time = time };
        frame = actionController.WriteFrame(frame);
        frames.Add(frame);
    }

    void FinalizeFrames() {
        actionController.Behavior.enabled = false;
        StreamWriter writer = new StreamWriter(savepath.framespath);
        for(int i = 0; i < frames.Count; i++) {
            writer.WriteLine(JsonUtility.ToJson(frames[i]));
        }
        writer.Close();
    }
}

[Serializable]
public class Frame {
    public int frame;
    public float time;
    public float target_posX;
    public float target_posY;
    public float target_posZ;
    public float target_rotX;
    public float target_rotY;
    public float target_rotZ;
    public float target_rotW;
    public float hand_posX;
    public float hand_posY;
    public float hand_posZ;
    public int hand_state;
    public int target_state;
}