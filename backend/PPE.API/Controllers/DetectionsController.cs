using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PPE.API.Data;
using PPE.API.Models;
using Microsoft.AspNetCore.SignalR;
using PPE.API.Hubs;

namespace PPE.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DetectionsController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IHubContext<DetectionHub> _hubContext;


        public DetectionsController(AppDbContext context , IHubContext<DetectionHub> hubContext)
        {
            _context = context;
            _hubContext = hubContext;
        }
private const string ViolationClassName = "head";
private const float ConfidenceThreshold = 0.6f;
private const int ViolationWindowSeconds = 5;
private const int LookbackSeconds = 30; // fenêtre large pour retrouver le début du "streak"

[HttpPost]
public async Task<IActionResult> PostDetection([FromBody] Detection detection)
{
    detection.Timestamp = DateTime.UtcNow;
    _context.Detections.Add(detection);
    await _context.SaveChangesAsync();

    if (detection.ClassName == ViolationClassName && detection.Confidence >= ConfidenceThreshold)
    {
        var lookbackStart = DateTime.UtcNow.AddSeconds(-LookbackSeconds);

        // On cherche la détection "head" la plus ANCIENNE dans une fenêtre large
        var earliestRecentHead = await _context.Detections
            .Where(d => d.ClassName == ViolationClassName
                        && d.Confidence >= ConfidenceThreshold
                        && d.Timestamp >= lookbackStart)
            .OrderBy(d => d.Timestamp)
            .FirstOrDefaultAsync();

        bool isContinuousViolation = earliestRecentHead != null
            && (DateTime.UtcNow - earliestRecentHead.Timestamp).TotalSeconds >= ViolationWindowSeconds;

        bool hasActiveViolation = await _context.Violations
            .AnyAsync(v => !v.Resolved);

        if (isContinuousViolation && !hasActiveViolation)
        {
            var violation = new Violation
            {
                DetectionId = detection.Id,
                AlertSent = false,
                Resolved = false,
                CreatedAt = DateTime.UtcNow
            };
            _context.Violations.Add(violation);
            await _context.SaveChangesAsync();

            await _hubContext.Clients.All.SendAsync("ReceiveDetection", detection);
        }
    }

    return Ok(detection);
}


        [HttpGet]
        public async Task<IActionResult> GetDetections()
        {
            var detections = await _context.Detections
                .OrderByDescending(d => d.Timestamp)
                .Take(50)
                .ToListAsync();

            return Ok(detections);
        }
    }
}