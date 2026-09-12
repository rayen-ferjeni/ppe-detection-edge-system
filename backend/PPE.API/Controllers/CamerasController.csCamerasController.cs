using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PPE.API.Data;
using PPE.API.Models;

namespace PPE.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class CamerasController : ControllerBase
    {
        private readonly AppDbContext _context;

        public CamerasController(AppDbContext context)
        {
            _context = context;
        }

        [HttpPost]
        public async Task<IActionResult> RegisterCamera([FromBody] Camera camera)
        {
            camera.LastSeenAt = DateTime.UtcNow;
            _context.Cameras.Add(camera);
            await _context.SaveChangesAsync();
            return Ok(camera);
        }

        [HttpGet]
        public async Task<IActionResult> GetCameras()
        {
            var cameras = await _context.Cameras.ToListAsync();
            return Ok(cameras);
        }
    }
}