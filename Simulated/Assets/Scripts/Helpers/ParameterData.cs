using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "ParameterData", menuName = "Scriptable Objects/Parameter Data")]
public class ParameterData : ScriptableObject
{
    public float reachSpeed = 1f;
    public float retractSpeed = 1f;
    public float throwSpeed = 1f;
    public float hitSpeed = 1f;

    [SerializeField] float pickSpeedMin = 1f;
    [SerializeField] float pickSpeedMax = 2f;
    [HideInInspector] public float pickSpeed;

    [SerializeField] float putSpeedMin = 1f;
    [SerializeField] float putSpeedMax = 2f;
    [HideInInspector] public float putSpeed;

    [SerializeField] float pushSpeedMin = 1f;
    [SerializeField] float pushSpeedMax = 2f;
    [HideInInspector] public float pushSpeed;

    [SerializeField] float throwForceMin = 100f;
    [SerializeField] float throwForceMax = 200f;
    [HideInInspector] public float throwForce;

    [SerializeField] float hitForceMin = 100f;
    [SerializeField] float hitForceMax = 200f;
    [HideInInspector] public float hitForce;

    [SerializeField] float massMin = 1f;
    [SerializeField] float massMax = 2f;
    [HideInInspector] public float mass;

    [SerializeField] float dragMin = 0f;
    [SerializeField] float dragMax = 1f;
    [HideInInspector] public float drag;

    [SerializeField] float angularDragMin = 0f;
    [SerializeField] float angularDragMax = 1f;
    [HideInInspector] public float angularDrag;

    [SerializeField] float dynamicFrictionMin = 0f;
    [SerializeField] float dynamicFrictionMax = 1f;
    [HideInInspector] public float dynamicFriction;

    [SerializeField] float staticFrictionMin = 0f;
    [SerializeField] float staticFrictionMax = 1f;
    [SerializeField] float staticFrictionMaxDelta = .3f;
    [HideInInspector] public float staticFriction;

    [SerializeField] float bouncinessMin = 0f;
    [SerializeField] float bouncinessMax = 1f;
    [HideInInspector] public float bounciness;

    [HideInInspector] public int activeMesh;

    public void Initialize() {
        pickSpeed = Random.Range(pickSpeedMin, pickSpeedMax);
        putSpeed = Random.Range(putSpeedMin, putSpeedMax);
        pushSpeed = Random.Range(pushSpeedMin, pushSpeedMax);
        throwForce = Random.Range(throwForceMin, throwForceMax);
        hitForce = Random.Range(hitForceMin, hitForceMax);
        mass = Random.Range(massMin, massMax);
        drag = Random.Range(dragMin, dragMax);
        angularDrag = Random.Range(angularDragMin, angularDragMax);
        activeMesh = Random.Range(0, 4);
        dynamicFriction = Random.Range(dynamicFrictionMin, dynamicFrictionMax);
        staticFriction = Mathf.Clamp(Random.Range(staticFrictionMin, staticFrictionMax), dynamicFriction - staticFrictionMaxDelta, dynamicFriction + staticFrictionMaxDelta);
        bounciness = Random.Range(bouncinessMin, bouncinessMax);
    }
}
