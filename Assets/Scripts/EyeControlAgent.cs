using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;

public class EyeControlAgent : Agent
{
    /// <summary>
    /// Define the index related to each posible discrete action rotation direction.
    /// </summary>
    enum CameraRotationActions : int
    {
        Negative = 0,
        Nothing = 1,
        Positive = 2,
    }

    /// <summary>
    /// Agent eyes strcut, holding the left and right eyes' state.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    struct AgentEyes<T>
    {
        public T Left;
        public T Right;
    }

    /// <summary>
    /// Agent state struct, holding the previous and current agent's state.
    /// </summary>
    /// <typeparam name="T">The type for the state.</typeparam>
    struct AgentState<T>
    {
        public T Current;
        public T Previous;

        /// <summary>
        /// Update previous agent state with current agent state.
        /// </summary>
        public void Update()
        {
            Previous = Current;
        }
    }

    #region Parameters    
    [Header("Eye control parameters")]
	[Tooltip("Target for the agent to look for with both eyes (cameras).")]
	public GameObject target;

	[Tooltip("Camera used for rendering the left side view.")]
	public Camera leftEyeCamera;

	[Tooltip("Camera used for rendering the right side view.")]
	public Camera rightEyeCamera;

	[Tooltip("Angles used to limit left eye camera rotation.")]
	public Vector2 leftEyeRestrictionAngle;

	[Tooltip("Angles used to limit right eye camera rotation.")]
	public Vector2 rightEyeRestrictionAngle;

	[Tooltip("Controls the speed at wich the agent is able to rotate each eye, in degrees per second.")]
	[Range(10.0f, 180.0f)]
	public float eyeRotationSpeed;

    [Tooltip("Controls the amount of time the target agent showld keep eyesight on the target before goal.")]
    [Range(0.0f, 10.0f)]
    public float eyeOnTargetTime;

    [Tooltip("Whether the agent should be given the eyes' angles as observations.")]
    public bool useCameraPositionObservations;

	[Header("On inference")]
	[Tooltip("Time between desicions on inference mode.")]
	[Range(0.05f, 0.5f)]
	public float timeBetweenDecisionsAtInference;

	/// <summary>
	/// Reward coeficient calculated before the start of training.
	/// Adjust rewards to ensure on goal completion agent's reward at least breck even.
	/// </summary>
	float m_RewardCoeficient;

	/// <summary>
	/// Time since last desicion was taken in inference mode.
	/// </summary>
	float m_TimeSinceDecision;

	/// <summary>
	/// The amount of time the agent has stayed on target.
	/// </summary>
	float m_TimeOnTarget;

    /// <summary>
    /// Academy training paramters.
    /// </summary>
    EnvironmentParameters m_ResetParameters;

    /// <summary>
    /// Goal vector sensor component, dedicated to goal signals.
    /// </summary>
    VectorSensorComponent m_GoalSensor;

    /// <summary>
    /// The target's mesh renderer.
    /// </summary>
    MeshRenderer m_TargetMeshRenderer;

    /// <summary>
    /// Wheter the camera is fully focused on the target or not.
    /// </summary>
    AgentEyes<AgentState<bool>> m_ViewOnTarget;

    /// <summary>
    /// Wheter the camera is partially focused on the target or not.
    /// </summary>
    AgentEyes<AgentState<bool>> m_ViewPartiallyOnTarget;

    /// <summary>
    /// Position of the target on the eye's viewport.
    /// </summary>
    AgentEyes<AgentState<Vector3>> m_ViewportTargetPosition;
    #endregion

    #region On Start Methods
    /// <summary>
    /// Initialice agent parameters and set enviroment.
    /// </summary>
    public override void Initialize()
    {
        // If on inference mode or heuristic.
        if (!Academy.Instance.IsCommunicatorOn)
        {
            // Set the max step to zero.
            this.MaxStep = 0;
        }

        // Calculate reward coeficient based on max step, equivalent to the amount of seconds for full episode.
        m_RewardCoeficient = MaxStep / 50;

        // Get the goal vector sensor component.
        m_GoalSensor = this.GetComponent<VectorSensorComponent>();

        // Get enviroment parameters from the academy parameters.
        m_ResetParameters = Academy.Instance.EnvironmentParameters;

        // Get target's mesh renderer.
        m_TargetMeshRenderer = target.GetComponent<MeshRenderer>();

        // Clear the eye sight timer.
        m_TimeOnTarget = 0.0f;

        // Set the agent.
        SetAgent();

        // Set the target.
        SetTarget();

        // Apply the current curriculum parameteres.
        ApplyCurriculum();
    }

    /// <summary>
	/// For every episode, set the agent and the target.
	/// </summary>
	public override void OnEpisodeBegin()
    {
        // Set the previous state for if the agent has either eye on the target to false.
        m_ViewOnTarget.Left.Previous = false;
        m_ViewOnTarget.Right.Previous = false;
        m_ViewPartiallyOnTarget.Left.Previous = false;
        m_ViewPartiallyOnTarget.Right.Previous = false;

        // Calculate the target's center Viewport on the cameras' viewports.
        m_ViewportTargetPosition.Left.Previous = leftEyeCamera.WorldToViewportPoint(m_TargetMeshRenderer.bounds.center);
        m_ViewportTargetPosition.Right.Previous = rightEyeCamera.WorldToViewportPoint(m_TargetMeshRenderer.bounds.center);

        // Clear the eye sight timer.
        m_TimeOnTarget = 0.0f;

        // Set the agent.
        SetAgent();

        // Set the target.
        SetTarget();

        // Apply the current curriculum parameteres.
        ApplyCurriculum();
    }

    /// <summary>
	/// Randomice agent eyes' orientation.
	/// </summary>
	void SetAgent()
    {
        // Rotate left camera to random angles in the x and y planes.
        leftEyeCamera.transform.Rotate(Vector3.right, Random.Range(-leftEyeRestrictionAngle.x, leftEyeRestrictionAngle.x));
        leftEyeCamera.transform.Rotate(Vector3.up, Random.Range(-leftEyeRestrictionAngle.y, leftEyeRestrictionAngle.y));

        // Rotate right camera to random angles in the x and y planes.
        rightEyeCamera.transform.Rotate(Vector3.right, Random.Range(-rightEyeRestrictionAngle.x, rightEyeRestrictionAngle.x));
        rightEyeCamera.transform.Rotate(Vector3.up, Random.Range(-rightEyeRestrictionAngle.y, rightEyeRestrictionAngle.y));
    }

    /// <summary>
    /// Randomice target position on backgrownd.
    /// </summary>
    void SetTarget()
    {
        // Randomice position in y and z.
        target.transform.localPosition = new Vector3(5, Random.Range(-4, 4), Random.Range(-4, 4));
    }

    /// <summary>
    /// Apply curriculum parameters.
    /// </summary>
    void ApplyCurriculum()
    {
        // Set cameras' field of view.
        leftEyeCamera.focalLength = m_ResetParameters.GetWithDefault("focal_length", 25f);
        rightEyeCamera.focalLength = m_ResetParameters.GetWithDefault("focal_length", 25f);
    }
    #endregion

    #region On Update Methods
    /// <summary>
	/// Update method called once per frame.
	/// </summary>
	public void FixedUpdate()
    {
        // If not on inference mode or heuristic.
        if (Academy.Instance.IsCommunicatorOn)
        {
            // Manually request a desicion.
            RequestDecision();
        }

        // If on inference mode or heuristic.
        else
        {
            // If the time required between desicions has passed.
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                // Manually request a desicion.
                RequestDecision();

                // Reset time between decisions.
                m_TimeSinceDecision = 0f;
            }

            // If the time between desicions hasn't passed.
            else
            {
                // Add the time between fixed updates to the time between desicions.
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }

    /// <summary>
    /// Add vector observations if the useVectorObservations flag is set.
    /// Use camera rotations in the x and y planes as observations.
    /// Also use camera restriction angles.
    /// </summary>
    /// <param name="sensor">Vector sensor to add observations to.</param>
    public override void CollectObservations(VectorSensor sensor)
    {
        // If the agent should use camera position observations.
        if (useCameraPositionObservations)
        {
            // Add left eye rotation as observations.
            sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.x) / 180f);
            sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.y) / 180f);

            // Add right eye rotation as observations.
            sensor.AddObservation(WrapAngle(rightEyeCamera.transform.localRotation.eulerAngles.x) / 180f);
            sensor.AddObservation(WrapAngle(rightEyeCamera.transform.localRotation.eulerAngles.y) / 180f);

            // Add left eye restriction angles as observations.
            sensor.AddObservation(leftEyeRestrictionAngle.x / 180f);
            sensor.AddObservation(leftEyeRestrictionAngle.y / 180f);

            // Add right eye restriction angles as observations.
            sensor.AddObservation(rightEyeRestrictionAngle.x / 180f);
            sensor.AddObservation(rightEyeRestrictionAngle.y / 180f);
        }

        // Calculate the target's center Viewport on the cameras' viewports.
        m_ViewportTargetPosition.Left.Current = leftEyeCamera.WorldToViewportPoint(m_TargetMeshRenderer.bounds.center);
        m_ViewportTargetPosition.Right.Current = rightEyeCamera.WorldToViewportPoint(m_TargetMeshRenderer.bounds.center);

        // If the left eye had the target partially within view.
        if (m_ViewPartiallyOnTarget.Left.Previous)
        {
            // Add boolean flag as observation, for target out of bounds.
            m_GoalSensor.GetSensor().AddObservation(false);

            // Add relative target's x and y screen position as observations.
            m_GoalSensor.GetSensor().AddObservation(m_ViewportTargetPosition.Left.Current.x);
            m_GoalSensor.GetSensor().AddObservation(m_ViewportTargetPosition.Left.Current.y);
            m_GoalSensor.GetSensor().AddObservation(0f);
        }

        // If the left eye did't have the target partially within view.
        else
        {
            // Add boolean flag as observation, for target out of bounds.
            m_GoalSensor.GetSensor().AddObservation(true);

            // Add out of bounds observation.
            m_GoalSensor.GetSensor().AddObservation(-1);
            m_GoalSensor.GetSensor().AddObservation(-1);
            m_GoalSensor.GetSensor().AddObservation(Random.Range(-1f, 1f));
        }

        // If the right eye had the target partially within view.
        if (m_ViewPartiallyOnTarget.Right.Previous)
        {
            // Add boolean flag as observation, for target out of bounds.
            m_GoalSensor.GetSensor().AddObservation(false);

            // Add relative target's x and y screen position as observations.
            m_GoalSensor.GetSensor().AddObservation(m_ViewportTargetPosition.Right.Current.x);
            m_GoalSensor.GetSensor().AddObservation(m_ViewportTargetPosition.Right.Current.y);
            m_GoalSensor.GetSensor().AddObservation(0f);
        }

        // If the right eye did't have the target partially within view.
        else
        {
            // Add boolean flag as observation, for target out of bounds.
            m_GoalSensor.GetSensor().AddObservation(true);

            // Add out of bounds observation.
            m_GoalSensor.GetSensor().AddObservation(-1);
            m_GoalSensor.GetSensor().AddObservation(-1);
            m_GoalSensor.GetSensor().AddObservation(Random.Range(-1f, 1f));
        }
    }

    /// <summary>
    /// Heuristic method of controlling agent motion.
    /// Set aggent eye motion from the arrow keys (X = horizontal arrows, Y + vertical arrows).
    /// </summary>
    /// <param name="actionsOut"></param>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Get descrete actions from the ActionBuffers.
        ActionSegment<int> descreteActions = actionsOut.DiscreteActions;

        // Set discrete actions for the X component of rotation to the Horizontal arrows.
        // Set discrete actions for the Y component of rotation to the Vertical arrows.
        descreteActions[0] = (int)Input.GetAxisRaw("Horizontal") + 1;
        descreteActions[1] = (int)Input.GetAxisRaw("Vertical") + 1;
        descreteActions[2] = (int)Input.GetAxisRaw("Horizontal") + 1;
        descreteActions[3] = (int)Input.GetAxisRaw("Vertical") + 1;
    }

    /// <summary>
    /// Receibe actions and process them for eye control.
    /// </summary>
    /// <param name="actionBuffers">Actions buffer to get actions from.</param>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Get left and right eye action buffers for the X and Y directions.
        var leftEyeActionX = actionBuffers.DiscreteActions[0];
        var leftEyeActionY = actionBuffers.DiscreteActions[1];
        var rightEyeActionX = actionBuffers.DiscreteActions[2];
        var rightEyeActionY = actionBuffers.DiscreteActions[3];

        // Rotate each camera by a given direction with the corresponding rotation action.
        RotateCamera(leftEyeCamera, Vector3.right, (CameraRotationActions)leftEyeActionX, eyeRotationSpeed * Time.deltaTime);
        RotateCamera(leftEyeCamera, Vector3.up, (CameraRotationActions)leftEyeActionY, eyeRotationSpeed * Time.deltaTime);
        RotateCamera(rightEyeCamera, Vector3.right, (CameraRotationActions)rightEyeActionX, eyeRotationSpeed * Time.deltaTime);
        RotateCamera(rightEyeCamera, Vector3.up, (CameraRotationActions)rightEyeActionY, eyeRotationSpeed * Time.deltaTime);

        // Clamp camera rotations to the corresponding restriction angles.
        ClampCameraRotation(leftEyeCamera, leftEyeRestrictionAngle.x, leftEyeRestrictionAngle.y);
        ClampCameraRotation(rightEyeCamera, rightEyeRestrictionAngle.x, rightEyeRestrictionAngle.y);

        // Check if the target is in the frustum for both the left and right cameras.
        m_ViewOnTarget.Left.Current = TargetInCameraFrustum(m_TargetMeshRenderer, leftEyeCamera);
        m_ViewOnTarget.Right.Current = TargetInCameraFrustum(m_TargetMeshRenderer, rightEyeCamera);

        // Check if the target is partially in the frustum for both the left and right cameras.
        m_ViewPartiallyOnTarget.Left.Current = TargetPartiallyInCameraFrustrum(m_TargetMeshRenderer, leftEyeCamera);
        m_ViewPartiallyOnTarget.Right.Current = TargetPartiallyInCameraFrustrum(m_TargetMeshRenderer, rightEyeCamera);

        // Reward the agent if the target comes into view of either eye.
        // Unreward the agent if the target comes out of view of either eye.
        AddReward((System.Convert.ToInt32(m_ViewOnTarget.Left.Current) - System.Convert.ToInt32(m_ViewOnTarget.Left.Previous)) * 0.2f * m_RewardCoeficient);
        AddReward((System.Convert.ToInt32(m_ViewOnTarget.Right.Current) - System.Convert.ToInt32(m_ViewOnTarget.Right.Previous)) * 0.2f * m_RewardCoeficient);

        // Reward the agent if the target comes partially into view of either eye.
        // Unreward the agent if the target comes out of view of either eye.
        AddReward((System.Convert.ToInt32(m_ViewPartiallyOnTarget.Left.Current) - System.Convert.ToInt32(m_ViewPartiallyOnTarget.Left.Previous)) * 0.1f * m_RewardCoeficient);
        AddReward((System.Convert.ToInt32(m_ViewPartiallyOnTarget.Right.Current) - System.Convert.ToInt32(m_ViewPartiallyOnTarget.Right.Previous)) * 0.1f * m_RewardCoeficient);

        // If the left eye was on target.
        if (m_ViewPartiallyOnTarget.Left.Previous & m_ViewPartiallyOnTarget.Left.Current)
        {
            // Reward the agent if the target comes closer to the center of view.
            // Unreward the agent if the target moves away from the center of view.
            AddReward((Math.Abs(m_ViewportTargetPosition.Left.Previous.x - 0.5f) - Math.Abs(m_ViewportTargetPosition.Left.Current.x - 0.5f)) * 0.1f * m_RewardCoeficient);
            AddReward((Math.Abs(m_ViewportTargetPosition.Left.Previous.y - 0.5f) - Math.Abs(m_ViewportTargetPosition.Left.Current.y - 0.5f)) * 0.1f * m_RewardCoeficient);
        }

        // If the right eye was on target.
        if (m_ViewPartiallyOnTarget.Right.Previous & m_ViewPartiallyOnTarget.Right.Current)
        {
            // Reward the agent if the target comes closer to the center of view.
            // Unreward the agent if the target comes closer to the center of view.
            AddReward((Math.Abs(m_ViewportTargetPosition.Right.Previous.x - 0.5f) - Math.Abs(m_ViewportTargetPosition.Right.Current.x - 0.5f)) * 0.1f * m_RewardCoeficient);
            AddReward((Math.Abs(m_ViewportTargetPosition.Right.Previous.y - 0.5f) - Math.Abs(m_ViewportTargetPosition.Right.Current.y - 0.5f)) * 0.1f * m_RewardCoeficient);
        }

        // If both eyes are on target.
        if (m_ViewOnTarget.Left.Current && m_ViewOnTarget.Right.Current)
        {
            // Add to the counter the amount of time that the agent has kept the target on view.
            m_TimeOnTarget += Time.deltaTime;
        }

        // If the agent doesn't have both eyes on the target.
        else
        {
            // Remove reward from agent the longer it takes to find the target.
            AddReward(-0.01f);

            // Clear the eye sight timer.
            m_TimeOnTarget = 0.0f;
        }

        // If the agent keeps the target on sight with both eyes for at least eyeOnTargetTime seconds.
        if (m_TimeOnTarget >= eyeOnTargetTime)
        {
            // Reward the agent for completing the episode.
            AddReward(0.4f * m_RewardCoeficient);

            // End the episode.
            EndEpisode();
        }

        // Set the previous agent's state.
        m_ViewOnTarget.Left.Update();
        m_ViewOnTarget.Right.Update();
        m_ViewPartiallyOnTarget.Left.Update();
        m_ViewPartiallyOnTarget.Right.Update();
        m_ViewportTargetPosition.Left.Update();
        m_ViewportTargetPosition.Right.Update();
    }
    #endregion

    #region Static Funcitons
    /// <summary>
    /// Rotate a given camera with a direction vector and the corresponding action.
    /// </summary>
    /// <param name="camera">The camera to rotate.</param>
    /// <param name="direction">The direction Vector3.</param>
    /// <param name="action">The corresponding action (positive rotation, negative rotation, or do nothing).</param>
    static void RotateCamera(Camera camera, Vector3 direction, CameraRotationActions action, float angle)
    {
        // On a given camera apply the control action.
        switch (action)
        {
            case CameraRotationActions.Nothing:
                // do nothing
                break;

            case CameraRotationActions.Positive:
                // Rotate camera in the given direction.
                camera.transform.Rotate(direction, angle);
                break;

            case CameraRotationActions.Negative:
                // Rotate camera in the oposite the given direction.
                camera.transform.Rotate(-direction, angle);
                break;
        }
    }

    /// <summary>
    /// Clamp the camera's x and y angles, by the given angles (restricted to: -angle, angle).
	/// The z angle is left as zero.
    /// </summary>
    /// <param name="camera">The camera to rotate.</param>
    /// <param name="angleX">The angle in X to clamp to.</param>
    /// <param name="angleY">The angle in Y to clamp to.</param>
    static void ClampCameraRotation(Camera camera, float angleX, float angleY)
    {
        // Get the clamped camara's rotation angles in the x, y, and z directions.
        float rotationX = Mathf.Clamp(WrapAngle(camera.transform.localRotation.eulerAngles.x), -angleX, angleX);
        float rotationY = Mathf.Clamp(WrapAngle(camera.transform.localRotation.eulerAngles.y), -angleY, angleY);

        // Fix the camera's rotation to the clamped angles (leave z angle as zero).
        camera.transform.localRotation = Quaternion.Euler(UnwrapAngle(rotationX), UnwrapAngle(rotationY), 0);
    }

    /// <summary>
    /// Pass Euler angles to a range of (-180, 180].
    /// </summary>
    /// <param name="angle">The angle to wrap.</param>
    /// <returns>The wraped angle in range (-180, 180].</returns>
    static float WrapAngle(float angle)
    {
		// Get the angle modulo 360.
        angle %= 360;

		// If the angle is greater than 180 degrees.
        if (angle > 180)
			// Return the angle as if starting from the other side of the cincunference.
            return angle - 360;

		// Otherwise, return the angle as is.
        return angle;
    }

    /// <summary>
    /// Pass angles in range (-180, 180] to Euler angles (range [0, 360)).
    /// </summary>
    /// <param name="angle">The angle to wrap.</param>
    /// <returns>The wraped angle in range [0, 360).</returns>
    static float UnwrapAngle(float angle)
    {
        // Get the angle modulo 360.
        angle %= 360;

        // If the angle is greater than zero.
        if (angle >= 0)
            // Return the angle as is.
            return angle;

        // Otherwise, return the angle as if starting from the other side of the cincunference. 
        return 360 + angle;
    }

    /// <summary>
    /// Check if the target's AABB from the target's bounds are outside the camera's frustum. 
    /// </summary>
    /// <param name="renderer">The target's mesh renderer.</param>
    /// <param name="camera">The camere to check for if the target is within its frustum.</param>
    /// <returns>True if the object is fully within the camera's frustum, false if otherwise.</returns>
    static bool TargetInCameraFrustum(Renderer renderer, Camera camera)
	{
        // Get the frustum planes for the left and right eye cameras.
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(camera);

		// Initialice condition on true. 
		bool isInFrustrum = true;

		// Iterate over each plane.
		for (int i = 0; i < planes.Length; i++)
		{
            // Flip the plane.
            Plane plane = planes[i].flipped;

            // Get the point on the edge of the bounding "elipsoid":
            // closest to the plane when the center is on the oposite side of the plane's normal,
            // and furthest to the plane when the center is on the same side of the plane's normal.
            Vector3 closestPoint = renderer.bounds.center + (plane.normal.normalized * renderer.bounds.extents.magnitude);

            // If the closest point is not on the same side as the plane's normal, "and" the result.
            isInFrustrum &= !plane.GetSide(closestPoint);
        }
		
		// Return if the target's bounds are in the camera's frustum planes.
		return isInFrustrum;
    }

    /// <summary>
    /// Check if the target's AABB from the target's bounds are outside the camera's frustum. 
    /// </summary>
    /// <param name="renderer">The target's mesh renderer.</param>
    /// <param name="camera">The camere to check for if the target is within its frustum.</param>
    /// <returns>True if the object is partially within the camera's frustum, false if otherwise.</returns>
    static bool TargetPartiallyInCameraFrustrum(Renderer renderer, Camera camera)
    {
        // Get the frustum planes for the left and right eye cameras.
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(camera);
		
        // Return if the AABB planes from the target's bounds are inside the camera's frustum planes.
        return GeometryUtility.TestPlanesAABB(planes, renderer.bounds);
    }
	#endregion
}
