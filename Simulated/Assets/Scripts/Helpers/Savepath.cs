using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

[CreateAssetMenu(fileName = "Savepath", menuName = "Scriptable Objects/Savepath")]
public class Savepath : ScriptableObject
{
    public string dirpath;
    public string metapath;
    public string framespath;

    public void Initialize() {
        string timestring = string.Format("{0}-{1}", DateTime.Now.Ticks.ToString(), System.Guid.NewGuid().ToString());
        SetPaths(timestring);
    }

    public void Initialize(string timestring) {
        SetPaths(timestring);
    }

    public void Cancel(bool imagesOnly = true) {
        if(!imagesOnly)
            Directory.Delete(dirpath, true);
        Directory.Delete(dirpath + "/images", true);
    }

    void SetPaths(string timestring) {
        string dataPath = Environment.GetEnvironmentVariable("SIMULATED_ROOT");
        dataPath += "/mini_raw";
        dirpath = string.Format("{0}/{1}", dataPath, timestring);        
        metapath = string.Format("{0}/meta.json", dirpath);
        framespath = string.Format("{0}/frames.json", dirpath);
        Debug.Log(string.Format("Directory set to to {0}", dirpath));
        Directory.CreateDirectory(dirpath);
    }
}
