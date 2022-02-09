using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class DemoCollection : MonoBehaviour
{
    [SerializeField] Transform target;
    [SerializeField] BoxCollider bounds;
    [SerializeField] string dirname = "BoxSlide";

    void Awake() {
        Physics.autoSimulation = false;
    }

    IEnumerator Start() {
        Vector3 force = Vector3.left * 500f;
        Rigidbody rigidBody = target.GetComponent<Rigidbody>();
        rigidBody.AddForce(force);
        float t = 0f;
        float delta = 1 / 60f;
        int frame = 0;
        List<DemoFrame> frames = new List<DemoFrame>();
        string dir = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        dir = Path.Combine(dir, "Simulated", dirname);
        string imgdir = Path.Combine(dir, "images");
        Directory.CreateDirectory(imgdir);
        while(t < 2f) {
            t += delta;
            string fpath = Path.Combine(imgdir, frame.ToString("D4") + ".png");
            ScreenCapture.CaptureScreenshot(fpath);
            Physics.Simulate(delta);
            Vector3[] corners = new Vector3[8];
            corners[0] = target.TransformPoint(bounds.center + new Vector3(-bounds.size.x, -bounds.size.y, -bounds.size.z) * .5f);
            corners[1] = target.TransformPoint(bounds.center + new Vector3(bounds.size.x, -bounds.size.y, -bounds.size.z) * .5f);
            corners[2] = target.TransformPoint(bounds.center + new Vector3(bounds.size.x, -bounds.size.y, bounds.size.z) * .5f);
            corners[3] = target.TransformPoint(bounds.center + new Vector3(-bounds.size.x, -bounds.size.y, bounds.size.z) * .5f);
            corners[4] = target.TransformPoint(bounds.center + new Vector3(-bounds.size.x, bounds.size.y, -bounds.size.z) * .5f);
            corners[5] = target.TransformPoint(bounds.center + new Vector3(bounds.size.x, bounds.size.y, -bounds.size.z) * .5f);
            corners[6] = target.TransformPoint(bounds.center + new Vector3(bounds.size.x, bounds.size.y, bounds.size.z) * .5f);
            corners[7] = target.TransformPoint(bounds.center + new Vector3(-bounds.size.x, bounds.size.y, bounds.size.z) * .5f);
            DemoFrame demoFrame = new DemoFrame() {
                frame = frame,
                time = t,
                target_posX = target.position.x,
                target_posY = target.position.y,
                target_posZ = target.position.z,
                corners = corners.ToList(),
                target_rotX = target.rotation.x,
                target_rotY = target.rotation.y,
                target_rotZ = target.rotation.z,
                target_rotW = target.rotation.w
            };
            frames.Add(demoFrame);
            frame++;
            yield return null;
        }
        DemoSequence sequence = new DemoSequence() {
            frames = frames,
            obj_name = target.gameObject.name,
            force = Vector3.left,
            mass = rigidBody.mass,
            drag = rigidBody.drag,
            angular_drag = rigidBody.angularDrag
        };
        string json = JsonUtility.ToJson(sequence);
        File.WriteAllText(Path.Combine(dir, "sequence.json"), json);
    }

    [Serializable]
    class DemoFrame
    {
        public int frame;
        public float time;
        public float target_posX;
        public float target_posY;
        public float target_posZ;
        public List<Vector3> corners;
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

    [Serializable]
    class DemoSequence 
    {
        public List<DemoFrame> frames;
        public string obj_name;
        public Vector3 force;
        public float mass;
        public float drag;
        public float angular_drag;
        public bool slide;
        public bool roll;
        public bool stack;
        public bool contain;
        public bool wgrasp;
        public bool bounce;
    }
}
