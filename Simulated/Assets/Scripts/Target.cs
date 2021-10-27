using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using BehaviorDesigner.Runtime;

public class Target : MonoBehaviour
{
    public event UnityAction OnHandEnter;
    public event UnityAction OnCounterEnter;

    public State state;

    public Rigidbody Rigidbody {
        get {
            if(rigidBody == null)
                rigidBody = GetComponent<Rigidbody>();
            return rigidBody;
        }
    }
    Rigidbody rigidBody;

    public bool Moving => Rigidbody.velocity.magnitude > 1e-5;

    [SerializeField] GameObject[] meshes;

    Transform tr;
    Vector3 prevPos;
    float velocity;

    public void OnValidate() {
        if(meshes.Length != 4)
            Debug.LogWarning("4 meshes expected");
    }
    
    void Awake() {
        tr = transform;
    }

    public void Initialize(int meshIndex, float mass, float drag, float angularDrag) {
        prevPos = tr.position = new Vector3(Random.Range(-.65f, .65f), .975f, Random.Range(-.65f, .65f));
        tr.rotation = Quaternion.Euler(0, Random.Range(0, 360f), 0);
        tr.SetParent(null);
        state = State.OnCounter;        
        SetMesh(meshIndex);
        Rigidbody.mass = mass;
        Rigidbody.drag = drag;
        Rigidbody.angularDrag = angularDrag;
        Rigidbody.velocity = Vector3.zero;
    }

    public void SetMesh(int meshIndex) {
        for(int i = 0; i < meshes.Length; i++)
            meshes[i].SetActive(false);
        meshes[meshIndex].SetActive(true);
    }

    void Update() {
        velocity = (tr.position - prevPos).magnitude;
        if(tr.parent != null)
            state = State.Held;
        else {
            if(Physics.Raycast(tr.position + Vector3.up * 2, Vector3.down, float.PositiveInfinity, LayerMask.GetMask("Counter"))) {
                state = State.OnCounter;
            } else {
                state = State.OnGround;
            }
        }
        prevPos = tr.position;
    }

    void OnCollisionEnter(Collision col) {
        if(col.gameObject.tag == "Hand")
            OnHandEnter?.Invoke();
        if(col.gameObject.tag == "Counter")
            OnCounterEnter?.Invoke();
    }

    public enum State {
        OnCounter,
        OnGround,
        Held
    }
}
