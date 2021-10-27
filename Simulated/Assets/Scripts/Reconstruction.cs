using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using SimpleFileBrowser;
using System;
using System.Linq;

public class Reconstruction : MonoBehaviour
{
    [SerializeField] Camera[] cameras;
    [SerializeField] Savepath savepath;
    [SerializeField] Transform hand;
    [SerializeField] Target target;
    [SerializeField] bool capture;

    bool busy;

    void Start() {
        FileBrowser.ShowLoadDialog(OnFilesSelected, null, FileBrowser.PickMode.Folders, true, Application.persistentDataPath, null, "Load", "Select");
    }

    void OnDisable() {
        if(busy) {
            savepath.Cancel();
        }
    }

    public void OnFilesSelected(string[] paths) {
        string[] timestrings = new string[paths.Length];
        for(int i = paths.Length - 1; i >= 0; i--) {
            string timestring = Path.GetFileName(paths[i]);
            timestrings[i] = timestring;
        }
        StartCoroutine(Run(timestrings));
    }

    IEnumerator Run(string[] timestrings) {
        float currentTime = Time.unscaledTime;
        float prevTime = Time.unscaledDeltaTime;
        Queue<float> timeQueue = new Queue<float>();
        for(int i = 0; i < timestrings.Length; i++) {
            prevTime = currentTime;
            Initialize(timestrings[i]);
            while(busy) yield return null;
            currentTime = Time.unscaledTime;
            Debug.Log(currentTime - prevTime);
            timeQueue.Enqueue(currentTime - prevTime);
            if(timeQueue.Count > 5) timeQueue.Dequeue();
            float avgTime = timeQueue.Average();
            float remaining = (timestrings.Length - i) * avgTime;
            TimeSpan t = TimeSpan.FromSeconds(remaining);
            Debug.Log("ETA: " + string.Format("{0:D2}d:{1:D2}h:{2:D2}m", t.Days, t.Hours, t.Minutes));
        }
    }

    void Initialize(string timestring) {
        savepath.Initialize(timestring);
        if(Directory.Exists(string.Format("{0}/images", savepath.dirpath))) {
            Debug.Log(string.Format("Skipping {0}, images directory already exists", savepath.dirpath));
            return;
        }
        for(int i = 0; i < cameras.Length; i++)
            Directory.CreateDirectory(string.Format("{0}/images/{1}", savepath.dirpath, cameras[i].gameObject.name));
        string json = File.ReadAllText(savepath.metapath);
        ParameterData data = ScriptableObject.CreateInstance<ParameterData>();
        JsonUtility.FromJsonOverwrite(json, data);
        target.SetMesh(data.activeMesh);
        StartCoroutine(Render(timestring));
    }

    IEnumerator Render(string timestring) {
        busy = true;
        for(int i = 0; i < cameras.Length; i++) {
            StreamReader reader = new StreamReader(savepath.framespath);
            Frame frame;
            string line;
            while((line = reader.ReadLine()) != null) {
                frame = JsonUtility.FromJson<Frame>(line);
                hand.position = new Vector3(frame.hand_posX, frame.hand_posY, frame.hand_posZ);
                target.transform.position = new Vector3(frame.target_posX, frame.target_posY, frame.target_posZ);
                target.transform.rotation = new Quaternion(frame.target_rotX, frame.target_rotY, frame.target_rotZ, frame.target_rotW);
                if(capture)
                    Capture(cameras[i], frame.frame);
                yield return null;
            }
            reader.Close();
        }        
        busy = false;
    }

    void Capture(Camera cam, int frame) {
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        cam.Render();

        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;

        byte[] bytes = image.EncodeToPNG();

        string path = string.Format("{0}/images/{1}/{2}.png", savepath.dirpath, cam.gameObject.name, frame.ToString("D5"));
        File.WriteAllBytes(path, bytes);

        Destroy(image);
    }
}
